"""SQLite alert store with audit log for acknowledge / dismiss / escalate.

All alert handling actions are attributed to a human operator session.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
                    metadata_json TEXT NOT NULL DEFAULT '{}'
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

                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
                CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_alert ON audit_log(alert_id);
                """
            )

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
    ) -> Alert:
        alert = Alert(
            id=str(uuid.uuid4()),
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
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO alerts (
                    id, created_at, source_label, category, risk_level, confidence,
                    rationale, frame_index, timestamp_sec, snapshot_path, status,
                    detections_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Alert:
        return Alert(
            id=row["id"],
            created_at=row["created_at"],
            source_label=row["source_label"],
            category=row["category"],
            risk_level=row["risk_level"],
            confidence=row["confidence"],
            rationale=row["rationale"],
            frame_index=row["frame_index"],
            timestamp_sec=row["timestamp_sec"],
            snapshot_path=row["snapshot_path"],
            status=row["status"],
            detections_json=row["detections_json"],
            metadata_json=row["metadata_json"],
        )
