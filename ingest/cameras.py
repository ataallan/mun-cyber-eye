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

from vision.scene_context import validate_place_type

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
                """
            )
            self._ensure_column(conn, "cameras", "place_type", "TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection, table: str, column: str, decl: str
    ) -> None:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

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
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO cameras (
                    id, name, location_label, source_type, uri, enabled,
                    sample_fps, sample_interval, notes, created_at, updated_at,
                    last_seen_at, last_error, place_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
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
        )
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE cameras SET
                    name = ?, location_label = ?, source_type = ?, uri = ?,
                    enabled = ?, sample_fps = ?, sample_interval = ?, notes = ?,
                    updated_at = ?, place_type = ?
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


def camera_from_form(form: Any, *, existing: Optional[Camera] = None) -> dict[str, Any]:
    """Parse a Flask form into CameraStore create/update kwargs (no raw dump)."""
    payload: dict[str, Any] = {
        "name": (form.get("name") or "").strip(),
        "location_label": (form.get("location_label") or "").strip(),
        "source_type": (form.get("source_type") or "file").strip().lower(),
        "notes": (form.get("notes") or "").strip(),
        "enabled": (form.get("enabled") or "0") == "1",
        "place_type": (form.get("place_type") or "").strip(),
    }
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
