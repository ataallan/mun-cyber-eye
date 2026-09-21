"""SQLite alert store with audit log, recipients, and delivery tracking.

All alert handling actions are attributed to a human operator session.
Outbound delivery never invents success — missing keys stay queued/undelivered.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional

from .schema import (
    format_frame_time,
    recommended_human_action,
    severity_from_risk,
    short_rationale,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(normalize_email(email)))


@dataclass
class Alert:
    id: str
    created_at: str
    source_label: str
    category: str
    risk_level: str
    confidence: float
    rationale: str
    frame_index: int
    timestamp_sec: float
    snapshot_path: Optional[str]
    status: str = "open"  # open | acknowledged | dismissed | escalated
    detections_json: str = "[]"
    metadata_json: str = "{}"
    severity: str = ""
    location_label: str = ""
    camera_id: str = ""
    frame_time: str = ""
    short_rationale: str = ""
    recommended_human_action: str = ""
    correlation_id: str = ""
    delivery_status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        try:
            d["detections"] = json.loads(self.detections_json)
        except json.JSONDecodeError:
            d["detections"] = []
        try:
            d["metadata"] = json.loads(self.metadata_json)
        except json.JSONDecodeError:
            d["metadata"] = {}
        return d


class AlertStore:
    """Persist alerts and an immutable-style audit trail of operator actions."""

    VALID_ACTIONS = {"acknowledge", "dismiss", "escalate", "reopen"}
    VALID_ROLES = {"admin", "operator"}
    VALID_DELIVERY = {
        "pending",
        "queued",
        "sent",
        "partial",
        "failed",
        "undelivered",
    }

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
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    category TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rationale TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    timestamp_sec REAL NOT NULL,
                    snapshot_path TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    detections_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    severity TEXT,
                    location_label TEXT,
                    camera_id TEXT,
                    frame_time TEXT,
                    short_rationale TEXT,
                    recommended_human_action TEXT,
                    correlation_id TEXT,
                    delivery_status TEXT NOT NULL DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (alert_id) REFERENCES alerts(id)
                );

                CREATE TABLE IF NOT EXISTS delivery_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    recipient TEXT,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    error TEXT,
                    provider_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (alert_id) REFERENCES alerts(id)
                );

                CREATE TABLE IF NOT EXISTS operators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT,
                    role TEXT NOT NULL DEFAULT 'operator',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
                CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_alert ON audit_log(alert_id);
                CREATE INDEX IF NOT EXISTS idx_delivery_alert ON delivery_log(alert_id);

                CREATE TABLE IF NOT EXISTS system_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_system_audit_created
                    ON system_audit(created_at DESC);
                """
            )
            self._migrate_alerts(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_delivery ON alerts(delivery_status)"
            )

    def _migrate_alerts(self, conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(alerts)")}
        additions = [
            ("severity", "TEXT"),
            ("location_label", "TEXT"),
            ("camera_id", "TEXT"),
            ("frame_time", "TEXT"),
            ("short_rationale", "TEXT"),
            ("recommended_human_action", "TEXT"),
            ("correlation_id", "TEXT"),
            ("delivery_status", "TEXT NOT NULL DEFAULT 'pending'"),
        ]
        for name, typ in additions:
            if name not in existing:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {name} {typ}")

    def create_alert(
        self,
        *,
        source_label: str,
        category: str,
        risk_level: str,
        confidence: float,
        rationale: str,
        frame_index: int,
        timestamp_sec: float,
        snapshot_path: Optional[str] = None,
        detections: Optional[List[dict]] = None,
        metadata: Optional[dict] = None,
        severity: Optional[str] = None,
        location_label: Optional[str] = None,
        camera_id: Optional[str] = None,
        frame_time: Optional[str] = None,
        short_rationale_text: Optional[str] = None,
        recommended_action: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Alert:
        alert_id = str(uuid.uuid4())
        alert = Alert(
            id=alert_id,
            created_at=_utc_now(),
            source_label=source_label,
            category=category,
            risk_level=risk_level,
            confidence=float(confidence),
            rationale=rationale,
            frame_index=int(frame_index),
            timestamp_sec=float(timestamp_sec),
            snapshot_path=snapshot_path,
            status="open",
            detections_json=json.dumps(detections or []),
            metadata_json=json.dumps(metadata or {}),
            severity=severity or severity_from_risk(risk_level),
            location_label=location_label or "",
            camera_id=camera_id or "",
            frame_time=frame_time or format_frame_time(timestamp_sec),
            short_rationale=short_rationale_text or short_rationale(rationale),
            recommended_human_action=recommended_action
            or recommended_human_action(category),
            correlation_id=correlation_id or alert_id,
            delivery_status="pending",
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO alerts (
                    id, created_at, source_label, category, risk_level, confidence,
                    rationale, frame_index, timestamp_sec, snapshot_path, status,
                    detections_json, metadata_json, severity, location_label,
                    camera_id, frame_time, short_rationale, recommended_human_action,
                    correlation_id, delivery_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.id,
                    alert.created_at,
                    alert.source_label,
                    alert.category,
                    alert.risk_level,
                    alert.confidence,
                    alert.rationale,
                    alert.frame_index,
                    alert.timestamp_sec,
                    alert.snapshot_path,
                    alert.status,
                    alert.detections_json,
                    alert.metadata_json,
                    alert.severity,
                    alert.location_label,
                    alert.camera_id,
                    alert.frame_time,
                    alert.short_rationale,
                    alert.recommended_human_action,
                    alert.correlation_id,
                    alert.delivery_status,
                ),
            )
            conn.execute(
                """
                INSERT INTO audit_log (alert_id, action, actor, note, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (alert.id, "created", "system", "Alert generated by risk engine", _utc_now()),
            )
        return alert

    def get(self, alert_id: str) -> Optional[Alert]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        return self._row_to_alert(row) if row else None

    def list_alerts(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        q = "SELECT * FROM alerts"
        params: list[Any] = []
        if status:
            q += " WHERE status = ?"
            params.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(q, params).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def apply_action(
        self,
        alert_id: str,
        action: str,
        actor: str,
        note: str = "",
    ) -> Alert:
        action = action.lower().strip()
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        status_map = {
            "acknowledge": "acknowledged",
            "dismiss": "dismissed",
            "escalate": "escalated",
            "reopen": "open",
        }
        new_status = status_map[action]

        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"Alert not found: {alert_id}")
            conn.execute(
                "UPDATE alerts SET status = ? WHERE id = ?",
                (new_status, alert_id),
            )
            conn.execute(
                """
                INSERT INTO audit_log (alert_id, action, actor, note, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (alert_id, action, actor, note or "", _utc_now()),
            )
            row = conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        return self._row_to_alert(row)

    def record_audit(self, alert_id: str, action: str, actor: str, note: str = "") -> None:
        with self._conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
            if not exists:
                raise KeyError(f"Alert not found: {alert_id}")
            conn.execute(
                """
                INSERT INTO audit_log (alert_id, action, actor, note, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (alert_id, action, actor, note or "", _utc_now()),
            )

    def record_system_audit(self, action: str, actor: str, note: str = "") -> None:
        """Record a console action that is not tied to a single alert (train/activate)."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO system_audit (action, actor, note, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (action, actor, note or "", _utc_now()),
            )

    def list_system_audit(self, limit: int = 25) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, action, actor, note, created_at
                FROM system_audit
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def audit_trail(self, alert_id: str) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, alert_id, action, actor, note, created_at
                FROM audit_log WHERE alert_id = ? ORDER BY id ASC
                """,
                (alert_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_delivery_status(self, alert_id: str, status: str) -> Alert:
        if status not in self.VALID_DELIVERY:
            raise ValueError(f"Invalid delivery status: {status}")
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"Alert not found: {alert_id}")
            conn.execute(
                "UPDATE alerts SET delivery_status = ? WHERE id = ?",
                (status, alert_id),
            )
            row = conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        return self._row_to_alert(row)

    def log_delivery(
        self,
        alert_id: str,
        *,
        channel: str,
        recipient: str = "",
        status: str,
        attempt: int = 1,
        error: str = "",
        provider_id: str = "",
    ) -> dict:
        with self._conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
            if not exists:
                raise KeyError(f"Alert not found: {alert_id}")
            conn.execute(
                """
                INSERT INTO delivery_log (
                    alert_id, channel, recipient, status, attempt, error,
                    provider_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_id,
                    channel,
                    recipient,
                    status,
                    int(attempt),
                    error or "",
                    provider_id or "",
                    _utc_now(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM delivery_log WHERE id = last_insert_rowid()"
            ).fetchone()
        return dict(row)

    def delivery_trail(self, alert_id: str) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, alert_id, channel, recipient, status, attempt,
                       error, provider_id, created_at
                FROM delivery_log WHERE alert_id = ? ORDER BY id ASC
                """,
                (alert_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_recipients(self, active_only: bool = False) -> List[dict]:
        q = "SELECT * FROM operators"
        if active_only:
            q += " WHERE active = 1"
        q += " ORDER BY email ASC"
        with self._conn() as conn:
            rows = conn.execute(q).fetchall()
        return [dict(r) for r in rows]

    def add_recipient(
        self,
        email: str,
        *,
        display_name: str = "",
        role: str = "operator",
        created_by: str = "system",
    ) -> dict:
        email = normalize_email(email)
        if not is_valid_email(email):
            raise ValueError("Invalid email address")
        role = (role or "operator").lower().strip()
        if role not in self.VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT * FROM operators WHERE email = ?", (email,)
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE operators
                    SET display_name = ?, role = ?, active = 1
                    WHERE email = ?
                    """,
                    (display_name or existing["display_name"] or "", role, email),
                )
                row = conn.execute(
                    "SELECT * FROM operators WHERE email = ?", (email,)
                ).fetchone()
                return dict(row)
            conn.execute(
                """
                INSERT INTO operators (
                    email, display_name, role, active, created_at, created_by
                ) VALUES (?, ?, ?, 1, ?, ?)
                """,
                (email, display_name or "", role, _utc_now(), created_by),
            )
            row = conn.execute(
                "SELECT * FROM operators WHERE email = ?", (email,)
            ).fetchone()
        return dict(row)

    def set_recipient_active(self, recipient_id: int, active: bool) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM operators WHERE id = ?", (recipient_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"Recipient not found: {recipient_id}")
            conn.execute(
                "UPDATE operators SET active = ? WHERE id = ?",
                (1 if active else 0, recipient_id),
            )
            row = conn.execute(
                "SELECT * FROM operators WHERE id = ?", (recipient_id,)
            ).fetchone()
        return dict(row)

    def stats(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM alerts GROUP BY status"
            ).fetchall()
        out = {"open": 0, "acknowledged": 0, "dismissed": 0, "escalated": 0, "total": 0}
        for r in rows:
            out[r["status"]] = r["n"]
            out["total"] += r["n"]
        return out

    def delivery_stats(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT delivery_status, COUNT(*) AS n FROM alerts GROUP BY delivery_status"
            ).fetchall()
        out = {
            "pending": 0,
            "queued": 0,
            "sent": 0,
            "partial": 0,
            "failed": 0,
            "undelivered": 0,
        }
        for r in rows:
            key = r["delivery_status"] or "pending"
            out[key] = r["n"]
        return out

    @staticmethod
    def _col(row: sqlite3.Row, name: str, default: Any = None) -> Any:
        try:
            value = row[name]
        except (IndexError, KeyError):
            return default
        return default if value is None else value

    def _row_to_alert(self, row: sqlite3.Row) -> Alert:
        risk_level = row["risk_level"]
        category = row["category"]
        rationale = row["rationale"]
        timestamp_sec = row["timestamp_sec"]
        alert_id = row["id"]
        return Alert(
            id=alert_id,
            created_at=row["created_at"],
            source_label=row["source_label"],
            category=category,
            risk_level=risk_level,
            confidence=row["confidence"],
            rationale=rationale,
            frame_index=row["frame_index"],
            timestamp_sec=timestamp_sec,
            snapshot_path=row["snapshot_path"],
            status=row["status"],
            detections_json=row["detections_json"],
            metadata_json=row["metadata_json"],
            severity=self._col(row, "severity") or severity_from_risk(risk_level),
            location_label=self._col(row, "location_label") or "",
            camera_id=self._col(row, "camera_id") or "",
            frame_time=self._col(row, "frame_time") or format_frame_time(timestamp_sec),
            short_rationale=self._col(row, "short_rationale") or short_rationale(rationale),
            recommended_human_action=self._col(row, "recommended_human_action")
            or recommended_human_action(category),
            correlation_id=self._col(row, "correlation_id") or alert_id,
            delivery_status=self._col(row, "delivery_status") or "pending",
        )
