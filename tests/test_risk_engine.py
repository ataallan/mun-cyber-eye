"""Tests for risk classification heuristics."""

from risk.engine import ActivityCategory, RiskEngine, RiskLevel
from vision.detector import Detection


def test_ordinary_person_only():
    engine = RiskEngine()
    result = engine.assess([Detection("person", 0.9, (0, 0, 10, 10))])
    assert result.category == ActivityCategory.ORDINARY
    assert result.risk_level == RiskLevel.LOW
    assert result.should_alert is False


def test_potential_fight():
    engine = RiskEngine()
    dets = [
        Detection("person", 0.9),
        Detection("person", 0.88),
        Detection("close_proximity", 0.8),
        Detection("rapid_motion", 0.82),
    ]
    result = engine.assess(dets)
    assert result.category == ActivityCategory.POTENTIAL_FIGHT
    assert result.risk_level in {RiskLevel.ELEVATED, RiskLevel.HIGH}
    assert result.should_alert is True
    assert result.confidence > 0.5


def test_potential_fall():
    engine = RiskEngine()
    dets = [
        Detection("person", 0.9),
        Detection("person_down", 0.81),
        Detection("horizontal_pose", 0.75),
    ]
    result = engine.assess(dets)
    assert result.category == ActivityCategory.POTENTIAL_FALL
    assert result.should_alert is True


def test_potential_weapon_object():
    engine = RiskEngine()
    dets = [
        Detection("person", 0.93),
        Detection("knife", 0.71),
        Detection("raised_object", 0.68),
    ]
    result = engine.assess(dets)
    assert result.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert result.risk_level in {RiskLevel.ELEVATED, RiskLevel.HIGH}
    assert result.should_alert is True
    assert "verification" in result.rationale.lower() or "human" in result.rationale.lower()


def test_empty_detections_ordinary():
    engine = RiskEngine()
    result = engine.assess([])
    assert result.category == ActivityCategory.ORDINARY
    assert result.should_alert is False
