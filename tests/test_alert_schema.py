"""Structured Phase 4 alert schema helpers and payload."""

import sqlite3

import pytest

from alerts.schema import (
    format_frame_time,
    recommended_human_action,
    severity_from_risk,
    short_rationale,
    structured_payload,
)
from alerts.store import AlertStore


@pytest.fixture
def store(tmp_path):
    return AlertStore(tmp_path / "schema.db")


def _create(store, **kwargs):
    defaults = dict(
        source_label="Authorized Camera — Test",
        category="potential_fight",
        risk_level="high",
        confidence=0.84,
        rationale="Movement patterns consistent with a potential confrontation. Review required.",
        frame_index=3,
        timestamp_sec=125.5,
        location_label="North Hall",
        camera_id="cam-07",
        correlation_id="corr-1",
    )
    defaults.update(kwargs)
    return store.create_alert(**defaults)


def test_severity_mapping():
    assert severity_from_risk("high") == "critical"
    assert severity_from_risk("elevated") == "warning"
    assert severity_from_risk("low") == "info"
    assert severity_from_risk("unknown") == "info"


def test_short_rationale_first_sentence():
    text = "First sentence. Second sentence continues."
    assert short_rationale(text) == "First sentence."


def test_frame_time_format():
    assert format_frame_time(125.5) == "00:02:05.50"
    assert format_frame_time(0) == "00:00:00.00"


def test_create_populates_phase4_fields(store):
    alert = _create(store)
    assert alert.severity == "critical"
    assert alert.location_label == "North Hall"
    assert alert.camera_id == "cam-07"
    assert alert.frame_time == "00:02:05.50"
    assert alert.short_rationale.startswith("Movement patterns")
    assert "Advisory only" in alert.recommended_human_action
    assert alert.correlation_id == "corr-1"
    assert alert.delivery_status == "pending"
    loaded = store.get(alert.id)
    assert loaded.severity == "critical"
    assert loaded.camera_id == "cam-07"


def test_structured_payload_contains_responder_fields(store):
    alert = _create(store)
    payload = structured_payload(alert)
    for key in (
        "severity",
        "category",
        "confidence",
        "location_label",
        "camera_id",
        "source",
        "frame_time",
        "snapshot_path",
        "short_rationale",
        "recommended_human_action",
        "correlation_id",
        "created_at",
        "delivery_status",
        "safety",
        "enforcement",
    ):
        assert key in payload
    assert payload["source"] == "Authorized Camera — Test"
    assert payload["safety"].startswith("AI detects")
    assert "no autonomous enforcement" in payload["enforcement"]


def test_advisory_action_is_not_enforcement():
    text = recommended_human_action("potential_weapon_object")
    assert "Advisory only" in text
    assert "not proof" in text.lower() or "not" in text.lower()


def test_legacy_db_gains_new_columns(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE alerts (
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
        """
    )
    conn.execute(
        """
        INSERT INTO alerts VALUES (
            'legacy-1', '2026-01-01T00:00:00Z', 'Cam', 'potential_fall',
            'elevated', 0.7, 'Possible fall.', 1, 1.0, NULL, 'open', '[]', '{}'
        )
        """
    )
    conn.commit()
    conn.close()

    store = AlertStore(db)
    loaded = store.get("legacy-1")
    assert loaded is not None
    assert loaded.severity == "warning"
    assert loaded.delivery_status == "pending"
    assert loaded.frame_time == "00:00:01.00"
    assert loaded.correlation_id == "legacy-1"
