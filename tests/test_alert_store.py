"""Tests for SQLite alert store and audit log."""

import pytest

from alerts.store import AlertStore


@pytest.fixture
def store(tmp_path):
    return AlertStore(tmp_path / "test_alerts.db")


def test_create_and_get(store):
    alert = store.create_alert(
        source_label="Authorized Camera — Test",
        category="potential_fight",
        risk_level="high",
        confidence=0.84,
        rationale="Test rationale requiring human review.",
        frame_index=3,
        timestamp_sec=1.5,
        snapshot_path=None,
        detections=[{"label": "person", "confidence": 0.9}],
    )
    loaded = store.get(alert.id)
    assert loaded is not None
    assert loaded.category == "potential_fight"
    assert loaded.status == "open"
    assert loaded.confidence == pytest.approx(0.84)


def test_acknowledge_audit(store):
    alert = store.create_alert(
        source_label="Authorized Camera — Test",
        category="potential_fall",
        risk_level="elevated",
        confidence=0.7,
        rationale="Possible fall — verify.",
        frame_index=1,
        timestamp_sec=0.5,
    )
    updated = store.apply_action(alert.id, "acknowledge", actor="operator", note="Reviewed feed")
    assert updated.status == "acknowledged"
    trail = store.audit_trail(alert.id)
    actions = [e["action"] for e in trail]
    assert "created" in actions
    assert "acknowledge" in actions
    assert trail[-1]["actor"] == "operator"


def test_dismiss_escalate_reopen(store):
    alert = store.create_alert(
        source_label="Authorized Camera — Test",
        category="potential_weapon_object",
        risk_level="high",
        confidence=0.72,
        rationale="Possible object — human verification required.",
        frame_index=5,
        timestamp_sec=2.0,
    )
    store.apply_action(alert.id, "escalate", actor="operator")
    assert store.get(alert.id).status == "escalated"
    store.apply_action(alert.id, "dismiss", actor="operator", note="False positive")
    assert store.get(alert.id).status == "dismissed"
    store.apply_action(alert.id, "reopen", actor="supervisor")
    assert store.get(alert.id).status == "open"


def test_list_and_stats(store):
    for i, level in enumerate(["low", "elevated", "high"]):
        store.create_alert(
            source_label="Cam",
            category="ordinary" if level == "low" else "potential_fight",
            risk_level=level,
            confidence=0.5 + i * 0.1,
            rationale="r",
            frame_index=i,
            timestamp_sec=float(i),
        )
    store.apply_action(store.list_alerts()[0].id, "acknowledge", actor="op")
    stats = store.stats()
    assert stats["total"] == 3
    assert stats["acknowledged"] == 1
    assert stats["open"] == 2


def test_invalid_action(store):
    alert = store.create_alert(
        source_label="Cam",
        category="ordinary",
        risk_level="low",
        confidence=0.5,
        rationale="r",
        frame_index=0,
        timestamp_sec=0.0,
    )
    with pytest.raises(ValueError):
        store.apply_action(alert.id, "delete", actor="op")
