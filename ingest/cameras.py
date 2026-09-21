"""SQLite camera registry for authorized file, RTSP, and webcam sources.

RTSP passwords are stored only when the operator pastes a full URI.
Prefer ``env:VAR_NAME`` secret references. UI helpers always mask credentials.
"""

from __future__ import annotations

import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from vision.scene_context import (
    CUSTOM_PLACE_SENTINEL,
    PlaceEntry,
    all_places,
    is_catalog_place,
    place_label_from_input,
    validate_place_type,
)

# Canonical demo-camera place stamps. Used for seed + empty-place backfill.
DEMO_PLACE_BACKFILL: dict[str, str] = {
    "demo-file-01": "gymnasium",
    "demo-rtsp-01": "street",
    "demo-corridor-01": "corridor_hallway",
    "demo-house-01": "house_interior",
    "demo-compound-01": "compound_courtyard",
    "demo-roam-01": "roam",
}

SOURCE_TYPES = frozenset({"file", "rtsp", "webcam"})
ENV_URI_PREFIX = "env:"
MOCK_URIS = frozenset({"", "mock", "MOCK", "mock://synthetic", "mock://demo"})

_URI_USERINFO_RE = re.compile(
    r"^((?:[a-zA-Z][a-zA-Z0-9+.-]*)://)([^/@]+):([^/@]+)@(.+)$"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_mock_uri(uri: str) -> bool:
    return (uri or "").strip() in MOCK_URIS or (uri or "").strip().lower() == "mock"


def is_env_ref(uri: str) -> bool:
    return (uri or "").strip().lower().startswith(ENV_URI_PREFIX)


def env_ref_name(uri: str) -> str:
    return (uri or "").strip()[len(ENV_URI_PREFIX) :]


def resolve_uri(uri: str) -> str:
    """Resolve ``env:VAR`` references. Missing secrets stay empty (honest)."""
    raw = (uri or "").strip()
    if is_env_ref(raw):
        name = env_ref_name(raw)
        if not name:
            return ""
        return (os.getenv(name) or "").strip()
    return raw


def mask_uri(uri: str) -> str:
    """Mask userinfo passwords for templates and API responses."""
    raw = (uri or "").strip()
    if not raw:
        return ""
    if is_env_ref(raw) or is_mock_uri(raw):
        return raw
    match = _URI_USERINFO_RE.match(raw)
    if match:
        scheme, user, _password, rest = match.groups()
        return f"{scheme}{unquote(user)}:****@{rest}"
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    if parts.username and parts.password is not None:
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        userinfo = f"{quote(parts.username, safe='')}:****@"
        netloc = f"{userinfo}{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return raw


def public_camera_dict(camera: "Camera") -> dict[str, Any]:
    data = asdict(camera)
    data["uri"] = mask_uri(camera.uri)
    data["uri_is_secret_ref"] = is_env_ref(camera.uri)
    return data


@dataclass
class Camera:
    id: str
    name: str
    location_label: str
    source_type: str
    uri: str
    enabled: bool
    sample_fps: float
    sample_interval: Optional[float]
    notes: str
    created_at: str
    updated_at: str
    last_seen_at: Optional[str] = None
    last_error: Optional[str] = None
    place_type: str = ""
    owner_user_id: str = ""
    owner_username: str = ""
    notify_email: str = ""

    def effective_fps(self) -> float:
        if self.sample_interval is not None and float(self.sample_interval) > 0:
            return max(0.1, 1.0 / float(self.sample_interval))
        return max(0.1, float(self.sample_fps or 2.0))

    def source_label(self) -> str:
        loc = (self.location_label or "").strip()
        name = (self.name or "Authorized Camera").strip()
        if loc and loc not in name:
            return f"{name} — {loc}"
        return name

    def to_public_dict(self) -> dict[str, Any]:
        return public_camera_dict(self)


class CameraStore:
    """Persist authorized cameras and health (last_seen / last_error)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cameras (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    location_label TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL CHECK (source_type IN ('file', 'rtsp', 'webcam')),
                    uri TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    sample_fps REAL NOT NULL DEFAULT 2.0,
                    sample_interval REAL,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_seen_at TEXT,
                    last_error TEXT,
                    place_type TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_cameras_enabled ON cameras(enabled);
                CREATE INDEX IF NOT EXISTS idx_cameras_updated ON cameras(updated_at DESC);

                CREATE TABLE IF NOT EXISTS custom_places (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "cameras", "place_type", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "cameras", "owner_user_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "cameras", "owner_username", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "cameras", "notify_email", "TEXT NOT NULL DEFAULT ''")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS camera_accounts (
                    camera_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (camera_id, user_id),
                    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_camera_accounts_user
                    ON camera_accounts(user_id);
                """
            )
            self._backfill_camera_accounts(conn)

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection, table: str, column: str, decl: str
    ) -> None:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    @staticmethod
    def _backfill_camera_accounts(conn: sqlite3.Connection) -> None:
        """Copy legacy single-owner columns into the many-to-many table."""
        rows = conn.execute(
            """
            SELECT id, owner_user_id, owner_username, created_at
            FROM cameras
            WHERE TRIM(COALESCE(owner_user_id, '')) != ''
            """
        ).fetchall()
        for row in rows:
            conn.execute(
                """
                INSERT OR IGNORE INTO camera_accounts
                    (camera_id, user_id, username, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["owner_user_id"],
                    row["owner_username"] or "",
                    row["created_at"],
                ),
            )

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM cameras").fetchone()
        return int(row["n"] if row else 0)

    def get(self, camera_id: str) -> Optional[Camera]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM cameras WHERE id = ?", (camera_id,)
            ).fetchone()
        return self._row_to_camera(row) if row else None

    def list_cameras(self, enabled_only: bool = False) -> List[Camera]:
        self.backfill_demo_place_types()
        q = "SELECT * FROM cameras"
        params: list[Any] = []
        if enabled_only:
            q += " WHERE enabled = 1"
        q += " ORDER BY name COLLATE NOCASE ASC"
        with self._conn() as conn:
            rows = conn.execute(q, params).fetchall()
        return [self._row_to_camera(r) for r in rows]

    def create(
        self,
        *,
        name: str,
        location_label: str = "",
        source_type: str,
        uri: str = "",
        enabled: bool = True,
        sample_fps: float = 2.0,
        sample_interval: Optional[float] = None,
        notes: str = "",
        camera_id: Optional[str] = None,
        place_type: str = "",
        owner_user_id: str = "",
        owner_username: str = "",
        notify_email: str = "",
    ) -> Camera:
        camera = Camera(
            id=camera_id or str(uuid.uuid4()),
            name=_require_name(name),
            location_label=(location_label or "").strip(),
            source_type=_validate_source_type(source_type),
            uri=(uri or "").strip(),
            enabled=bool(enabled),
            sample_fps=_validate_fps(sample_fps),
            sample_interval=_validate_interval(sample_interval),
            notes=(notes or "").strip(),
            created_at=_utc_now(),
            updated_at=_utc_now(),
            place_type=validate_place_type(place_type),
            owner_user_id=(owner_user_id or "").strip(),
            owner_username=(owner_username or "").strip(),
            notify_email=_normalize_notify_emails(notify_email),
        )
        self.remember_custom_place(camera.place_type, place_type)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO cameras (
                    id, name, location_label, source_type, uri, enabled,
                    sample_fps, sample_interval, notes, created_at, updated_at,
                    last_seen_at, last_error, place_type, owner_user_id,
                    owner_username, notify_email
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    camera.id,
                    camera.name,
                    camera.location_label,
                    camera.source_type,
                    camera.uri,
                    1 if camera.enabled else 0,
                    camera.sample_fps,
                    camera.sample_interval,
                    camera.notes,
                    camera.created_at,
                    camera.updated_at,
                    camera.place_type,
                    camera.owner_user_id,
                    camera.owner_username,
                    camera.notify_email,
                ),
            )
            if camera.owner_user_id:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO camera_accounts
                        (camera_id, user_id, username, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        camera.id,
                        camera.owner_user_id,
                        camera.owner_username,
                        camera.created_at,
                    ),
                )
        return camera

    def update(
        self,
        camera_id: str,
        *,
        name: Optional[str] = None,
        location_label: Optional[str] = None,
        source_type: Optional[str] = None,
        uri: Optional[str] = None,
        enabled: Optional[bool] = None,
        sample_fps: Optional[float] = None,
        sample_interval: Optional[float] = None,
        notes: Optional[str] = None,
        place_type: Optional[str] = None,
        owner_user_id: Optional[str] = None,
        owner_username: Optional[str] = None,
        notify_email: Optional[str] = None,
        clear_interval: bool = False,
    ) -> Camera:
        existing = self.get(camera_id)
        if existing is None:
            raise KeyError(f"Camera not found: {camera_id}")
        next_name = _require_name(name) if name is not None else existing.name
        next_type = (
            _validate_source_type(source_type)
            if source_type is not None
            else existing.source_type
        )
        next_uri = existing.uri if uri is None else (uri or "").strip()
        next_fps = (
            _validate_fps(sample_fps) if sample_fps is not None else existing.sample_fps
        )
        if clear_interval:
            next_interval = None
        elif sample_interval is not None:
            next_interval = _validate_interval(sample_interval)
        else:
            next_interval = existing.sample_interval
        updated = Camera(
            id=existing.id,
            name=next_name,
            location_label=(
                existing.location_label
                if location_label is None
                else location_label.strip()
            ),
            source_type=next_type,
            uri=next_uri,
            enabled=existing.enabled if enabled is None else bool(enabled),
            sample_fps=next_fps,
            sample_interval=next_interval,
            notes=existing.notes if notes is None else notes.strip(),
            created_at=existing.created_at,
            updated_at=_utc_now(),
            last_seen_at=existing.last_seen_at,
            last_error=existing.last_error,
            place_type=(
                existing.place_type
                if place_type is None
                else validate_place_type(place_type)
            ),
            owner_user_id=(
                existing.owner_user_id
                if owner_user_id is None
                else owner_user_id.strip()
            ),
            owner_username=(
                existing.owner_username
                if owner_username is None
                else owner_username.strip()
            ),
            notify_email=(
                existing.notify_email
                if notify_email is None
                else _normalize_notify_emails(notify_email)
            ),
        )
        if place_type is not None:
            self.remember_custom_place(updated.place_type, place_type)
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE cameras SET
                    name = ?, location_label = ?, source_type = ?, uri = ?,
                    enabled = ?, sample_fps = ?, sample_interval = ?, notes = ?,
                    updated_at = ?, place_type = ?, owner_user_id = ?,
                    owner_username = ?, notify_email = ?
                WHERE id = ?
                """,
                (
                    updated.name,
                    updated.location_label,
                    updated.source_type,
                    updated.uri,
                    1 if updated.enabled else 0,
                    updated.sample_fps,
                    updated.sample_interval,
                    updated.notes,
                    updated.updated_at,
                    updated.place_type,
                    updated.owner_user_id,
                    updated.owner_username,
                    updated.notify_email,
                    updated.id,
                ),
            )
        return updated

    def set_enabled(self, camera_id: str, enabled: bool) -> Camera:
        return self.update(camera_id, enabled=bool(enabled))

    def record_success(self, camera_id: str) -> Camera:
        if self.get(camera_id) is None:
            raise KeyError(f"Camera not found: {camera_id}")
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE cameras
                SET last_seen_at = ?, last_error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (now, now, camera_id),
            )
        loaded = self.get(camera_id)
        assert loaded is not None
        return loaded

    def record_error(self, camera_id: str, error: str) -> Camera:
        if self.get(camera_id) is None:
            raise KeyError(f"Camera not found: {camera_id}")
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE cameras
                SET last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                ((error or "Unknown ingest error")[:500], now, camera_id),
            )
        loaded = self.get(camera_id)
        assert loaded is not None
        return loaded

    def list_accounts_for_camera(self, camera_id: str) -> list[dict[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT user_id, username, created_at
                FROM camera_accounts
                WHERE camera_id = ?
                ORDER BY username COLLATE NOCASE ASC
                """,
                (camera_id,),
            ).fetchall()
        return [
            {
                "user_id": row["user_id"],
                "username": row["username"] or "",
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def accounts_by_camera(self) -> dict[str, list[dict[str, str]]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT camera_id, user_id, username, created_at
                FROM camera_accounts
                ORDER BY username COLLATE NOCASE ASC
                """
            ).fetchall()
        out: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            out.setdefault(row["camera_id"], []).append(
                {
                    "user_id": row["user_id"],
                    "username": row["username"] or "",
                    "created_at": row["created_at"],
                }
            )
        return out

    def list_camera_ids_for_user(self, user_id: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT camera_id FROM camera_accounts
                WHERE user_id = ?
                ORDER BY camera_id COLLATE NOCASE ASC
                """,
                (user_id,),
            ).fetchall()
        return [row["camera_id"] for row in rows]

    def set_accounts_for_camera(
        self, camera_id: str, accounts: list[tuple[str, str]]
    ) -> list[dict[str, str]]:
        """Replace every account linked to one camera (many accounts allowed)."""
        if self.get(camera_id) is None:
            raise KeyError(f"Camera not found: {camera_id}")
        now = _utc_now()
        unique: list[tuple[str, str]] = []
        seen: set[str] = set()
        for user_id, username in accounts:
            uid = (user_id or "").strip()
            if not uid or uid in seen:
                continue
            seen.add(uid)
            unique.append((uid, (username or "").strip()))
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM camera_accounts WHERE camera_id = ?", (camera_id,)
            )
            for user_id, username in unique:
                conn.execute(
                    """
                    INSERT INTO camera_accounts
                        (camera_id, user_id, username, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (camera_id, user_id, username, now),
                )
        self._sync_owner_columns(camera_id)
        return self.list_accounts_for_camera(camera_id)

    def set_cameras_for_user(
        self, user_id: str, username: str, camera_ids: list[str]
    ) -> list[str]:
        """Attach/detach many cameras on one account (account-centric)."""
        uid = (user_id or "").strip()
        if not uid:
            raise ValueError("User id is required.")
        wanted = []
        seen: set[str] = set()
        for camera_id in camera_ids:
            cid = (camera_id or "").strip()
            if not cid or cid in seen:
                continue
            if self.get(cid) is None:
                raise KeyError(f"Camera not found: {cid}")
            seen.add(cid)
            wanted.append(cid)
        existing = set(self.list_camera_ids_for_user(uid))
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM camera_accounts WHERE user_id = ?", (uid,)
            )
            for cid in wanted:
                conn.execute(
                    """
                    INSERT INTO camera_accounts
                        (camera_id, user_id, username, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cid, uid, (username or "").strip(), now),
                )
        affected = existing.union(wanted)
        for cid in affected:
            self._sync_owner_columns(cid)
        return self.list_camera_ids_for_user(uid)

    def _sync_owner_columns(self, camera_id: str) -> None:
        """Keep first linked account on the camera row for list/form display."""
        accounts = self.list_accounts_for_camera(camera_id)
        first = accounts[0] if accounts else None
        self.update(
            camera_id,
            owner_user_id=first["user_id"] if first else "",
            owner_username=first["username"] if first else "",
        )

    def health(self) -> dict[str, Any]:
        cameras = self.list_cameras()
        return {
            "total": len(cameras),
            "enabled": sum(1 for c in cameras if c.enabled),
            "with_error": sum(1 for c in cameras if c.last_error),
            "cameras": [c.to_public_dict() for c in cameras],
        }

    def backfill_demo_place_types(self) -> int:
        """Fill empty ``place_type`` on known demo cameras. Returns rows updated."""
        updated = 0
        now = _utc_now()
        with self._conn() as conn:
            for camera_id, place in DEMO_PLACE_BACKFILL.items():
                cur = conn.execute(
                    """
                    UPDATE cameras
                    SET place_type = ?, updated_at = ?
                    WHERE id = ? AND (place_type IS NULL OR TRIM(place_type) = '')
                    """,
                    (place, now, camera_id),
                )
                updated += int(cur.rowcount or 0)
        return updated

    def seed_demo_cameras(self, project_root: str | Path | None = None) -> List[Camera]:
        """Insert demo cameras (create-if-missing) and backfill empty place tags."""
        self.backfill_demo_place_types()
        existing_ids = {row["id"] for row in self._list_rows()}
        for spec in _demo_camera_specs(project_root):
            if spec["camera_id"] not in existing_ids:
                self.create(**spec)
        return self.list_cameras()

    def _list_rows(self) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return list(conn.execute("SELECT id FROM cameras").fetchall())

    def remember_custom_place(self, place_id: str, raw_label: str = "") -> Optional[PlaceEntry]:
        """Persist an operator-typed place so it appears in the next dropdown."""
        slug = validate_place_type(place_id or raw_label)
        if not slug or is_catalog_place(slug):
            return None
        display = place_label_from_input(raw_label or place_id, slug)
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO custom_places (id, display_name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    display_name = CASE
                        WHEN excluded.display_name != '' THEN excluded.display_name
                        ELSE custom_places.display_name
                    END
                """,
                (slug, display, now),
            )
        return PlaceEntry(slug, display, "custom")

    def list_custom_places(self) -> list[PlaceEntry]:
        self._harvest_camera_places()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, display_name FROM custom_places ORDER BY id COLLATE NOCASE ASC"
            ).fetchall()
        out: list[PlaceEntry] = []
        for row in rows:
            slug = row["id"]
            if not slug or is_catalog_place(slug):
                continue
            display = (row["display_name"] or "").strip() or place_label_from_input("", slug)
            out.append(PlaceEntry(slug, display, "custom"))
        return out

    def list_place_choices(self) -> list[PlaceEntry]:
        """Catalog places plus remembered custom ids (catalog first)."""
        seen = {p.id for p in all_places()}
        choices = list(all_places())
        for entry in self.list_custom_places():
            if entry.id not in seen:
                seen.add(entry.id)
                choices.append(entry)
        return choices

    def _harvest_camera_places(self) -> None:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT place_type FROM cameras WHERE TRIM(place_type) != ''"
            ).fetchall()
        for row in rows:
            slug = validate_place_type(row["place_type"] or "")
            if slug and not is_catalog_place(slug):
                self.remember_custom_place(slug, row["place_type"])

    @staticmethod
    def _row_to_camera(row: sqlite3.Row) -> Camera:
        return Camera(
            id=row["id"],
            name=row["name"],
            location_label=row["location_label"] or "",
            source_type=row["source_type"],
            uri=row["uri"] or "",
            enabled=bool(row["enabled"]),
            sample_fps=float(row["sample_fps"] or 2.0),
            sample_interval=row["sample_interval"],
            notes=row["notes"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_seen_at=row["last_seen_at"],
            last_error=row["last_error"],
            place_type=row["place_type"] if "place_type" in row.keys() else "",
            owner_user_id=_row_text(row, "owner_user_id"),
            owner_username=_row_text(row, "owner_username"),
            notify_email=_row_text(row, "notify_email"),
        )


def _demo_camera_specs(project_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Authorized demo / stub cameras with operator-requested place tags."""
    root = Path(project_root) if project_root else Path.cwd()
    demo_file = root / "sample_data" / "demo.mp4"
    file_uri = "MOCK"
    if demo_file.is_file():
        file_uri = str(Path("sample_data") / "demo.mp4")
    authorized = (
        "Authorized demo source. Uses MOCK synthetic frames unless a "
        "sample clip is present at sample_data/demo.mp4. Replace the "
        "URI with an authorized video file path for a real file run."
    )
    stub_note = (
        "Authorized demo stub (MOCK). Place type is a catalog stamp for "
        "operators — not a named venue or a determination of what happened."
    )
    return [
        {
            "camera_id": "demo-file-01",
            "name": "Demo Lab File",
            "location_label": "Demo Lab",
            "source_type": "file",
            "uri": file_uri,
            "enabled": True,
            "sample_fps": 2.0,
            "notes": authorized,
            "place_type": "gymnasium",
        },
        {
            "camera_id": "demo-rtsp-01",
            "name": "Authorized RTSP stub",
            "location_label": "Perimeter (stub)",
            "source_type": "rtsp",
            "uri": "env:RTSP_DEMO_URI",
            "enabled": False,
            "sample_fps": 2.0,
            "notes": (
                "Disabled by default. Set RTSP_DEMO_URI to an authorized RTSP "
                "URL (credentials stay in the environment). Do not point this "
                "at unauthorized streams."
            ),
            "place_type": "street",
        },
        {
            "camera_id": "demo-corridor-01",
            "name": "Corridor North",
            "location_label": "North corridor",
            "source_type": "file",
            "uri": "MOCK",
            "enabled": True,
            "sample_fps": 2.0,
            "notes": stub_note,
            "place_type": "corridor_hallway",
        },
        {
            "camera_id": "demo-house-01",
            "name": "House interior demo",
            "location_label": "House interior",
            "source_type": "file",
            "uri": "MOCK",
            "enabled": True,
            "sample_fps": 2.0,
            "notes": stub_note,
            "place_type": "house_interior",
        },
        {
            "camera_id": "demo-compound-01",
            "name": "Compound courtyard",
            "location_label": "Compound courtyard",
            "source_type": "file",
            "uri": "MOCK",
            "enabled": True,
            "sample_fps": 2.0,
            "notes": stub_note,
            "place_type": "compound_courtyard",
        },
        {
            "camera_id": "demo-roam-01",
            "name": "Roam / patrol cam",
            "location_label": "Patrol / multi-area",
            "source_type": "file",
            "uri": "MOCK",
            "enabled": True,
            "sample_fps": 2.0,
            "notes": stub_note,
            "place_type": "roam",
        },
    ]


def _require_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("Camera name is required.")
    if len(cleaned) > 120:
        raise ValueError("Camera name is too long.")
    return cleaned


def _validate_source_type(source_type: str) -> str:
    cleaned = (source_type or "").strip().lower()
    if cleaned not in SOURCE_TYPES:
        raise ValueError("Source type must be file, rtsp, or webcam.")
    return cleaned


def _validate_fps(sample_fps: float) -> float:
    try:
        value = float(sample_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_fps must be a number.") from exc
    if value <= 0:
        raise ValueError("sample_fps must be greater than 0.")
    return min(value, 30.0)


def _validate_interval(sample_interval: Optional[float]) -> Optional[float]:
    if sample_interval is None or sample_interval == "":
        return None
    try:
        value = float(sample_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_interval must be a number.") from exc
    if value <= 0:
        return None
    return value


def place_type_from_form(form: Any) -> str:
    """Read catalog select + optional custom text. Any non-empty value is allowed."""
    custom = (form.get("place_type_custom") or "").strip()
    choice = (form.get("place_type") or "").strip()
    if choice == CUSTOM_PLACE_SENTINEL:
        return custom
    if custom and not choice:
        return custom
    return choice


def _row_text(row: sqlite3.Row, key: str, default: str = "") -> str:
    if key not in row.keys():
        return default
    value = row[key]
    return default if value is None else str(value)


def _normalize_notify_emails(raw: str) -> str:
    """Store extra notify emails as a de-duplicated comma-separated list."""
    seen: set[str] = set()
    out: list[str] = []
    for part in (raw or "").replace(";", ",").split(","):
        email = part.strip().lower()
        if email and "@" in email and email not in seen:
            seen.add(email)
            out.append(email)
    return ", ".join(out)


def _form_list(form: Any, key: str) -> list[str]:
    if hasattr(form, "getlist"):
        values = form.getlist(key)
    else:
        raw = form.get(key) if hasattr(form, "get") else None
        values = raw if isinstance(raw, list) else ([raw] if raw else [])
    return [str(v).strip() for v in values if str(v).strip()]


def linked_accounts_from_form(
    form: Any, user_store: Any | None = None
) -> list[tuple[str, str]]:
    """Resolve many console accounts from camera-form checkboxes."""
    ids = _form_list(form, "account_user_ids")
    if not ids:
        one = (form.get("owner_user_id") or "").strip()
        if one:
            ids = [one]
    accounts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for uid in ids:
        if uid in seen:
            continue
        seen.add(uid)
        if user_store is None:
            accounts.append((uid, (form.get("owner_username") or "").strip()))
            continue
        user = user_store.get_by_id(uid)
        if user is None:
            raise ValueError("Selected camera account was not found.")
        accounts.append((user.id, user.username))
    return accounts


def owner_fields_from_form(form: Any, user_store: Any | None = None) -> dict[str, str]:
    """First linked account + extra notify emails (legacy owner columns)."""
    extra = _normalize_notify_emails(form.get("notify_email") or "")
    accounts = linked_accounts_from_form(form, user_store)
    if not accounts:
        return {
            "owner_user_id": "",
            "owner_username": "",
            "notify_email": extra,
        }
    user_id, username = accounts[0]
    return {
        "owner_user_id": user_id,
        "owner_username": username,
        "notify_email": extra,
    }


def camera_from_form(
    form: Any,
    *,
    existing: Optional[Camera] = None,
    user_store: Any | None = None,
) -> dict[str, Any]:
    """Parse a Flask form into CameraStore create/update kwargs (no raw dump)."""
    payload: dict[str, Any] = {
        "name": (form.get("name") or "").strip(),
        "location_label": (form.get("location_label") or "").strip(),
        "source_type": (form.get("source_type") or "file").strip().lower(),
        "notes": (form.get("notes") or "").strip(),
        "enabled": (form.get("enabled") or "0") == "1",
        "place_type": place_type_from_form(form),
    }
    payload.update(owner_fields_from_form(form, user_store))
    fps_raw = (form.get("sample_fps") or "").strip()
    if fps_raw:
        payload["sample_fps"] = float(fps_raw)
    elif existing is None:
        payload["sample_fps"] = 2.0
    interval_raw = (form.get("sample_interval") or "").strip()
    if interval_raw:
        payload["sample_interval"] = float(interval_raw)
    elif existing is not None:
        payload["clear_interval"] = True
    uri_raw = (form.get("uri") or "").strip()
    if uri_raw:
        payload["uri"] = uri_raw
    elif existing is None:
        if payload["source_type"] == "webcam":
            payload["uri"] = "0"
        elif payload["source_type"] == "file":
            payload["uri"] = "MOCK"
        else:
            payload["uri"] = ""
    return payload
