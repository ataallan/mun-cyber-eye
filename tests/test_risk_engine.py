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


def test_phase3_activity_label_fight():
    engine = RiskEngine()
    result = engine.assess(
        [
            Detection(
                "potential_fight",
                0.81,
                extras={"scores": {"potential_fight": 0.81, "ordinary": 0.1}},
            )
        ]
    )
    assert result.category == ActivityCategory.POTENTIAL_FIGHT
    assert result.should_alert is True
    assert result.risk_level in {RiskLevel.ELEVATED, RiskLevel.HIGH}
    assert "activity model" in result.rationale.lower()
    assert "confrontation" in result.rationale.lower()


def test_phase3_activity_label_ordinary_no_alert():
    engine = RiskEngine()
    result = engine.assess([Detection("ordinary", 0.9)])
    assert result.category == ActivityCategory.ORDINARY
    assert result.should_alert is False


def test_phase3_activity_label_overrides_heuristics():
    engine = RiskEngine()
    result = engine.assess(
        [
            Detection("ordinary", 0.88),
            Detection("knife", 0.91),
        ]
    )
    assert result.category == ActivityCategory.ORDINARY
    assert result.should_alert is False


def test_phase3_game_or_play_no_alert_by_default():
    engine = RiskEngine()
    result = engine.assess([Detection("game_or_play", 0.91)])
    assert result.category == ActivityCategory.GAME_OR_PLAY
    assert result.should_alert is False
    assert result.risk_level == RiskLevel.LOW
    assert "game or play" in result.rationale.lower()
    assert "not a fight" in result.rationale.lower()


def test_phase3_dance_no_alert_by_default():
    engine = RiskEngine()
    result = engine.assess([Detection("dance", 0.88)])
    assert result.category == ActivityCategory.DANCE
    assert result.should_alert is False
    assert "dance" in result.rationale.lower()
    assert "not a confrontation" in result.rationale.lower()


def test_phase3_confrontation_alias_alerts_as_fight():
    engine = RiskEngine()
    for alias in ("confrontation", "fight", "altercation"):
        result = engine.assess([Detection(alias, 0.8)])
        assert result.category == ActivityCategory.POTENTIAL_FIGHT
        assert result.should_alert is True


def test_phase3_game_or_dance_alert_when_env_enabled():
    engine = RiskEngine(alert_on_game_or_dance=True)
    play = engine.assess([Detection("game_or_play", 0.9)])
    dance = engine.assess([Detection("dancing", 0.86)])
    assert play.should_alert is True
    assert dance.should_alert is True
    assert play.category == ActivityCategory.GAME_OR_PLAY
    assert dance.category == ActivityCategory.DANCE


def test_should_alert_matrix_defaults():
    engine = RiskEngine()
    expected = {
        "ordinary": False,
        "game_or_play": False,
        "dance": False,
        "potential_fight": True,
        "potential_fall": True,
        "potential_weapon_object": True,
    }
    for label, should_alert in expected.items():
        result = engine.assess([Detection(label, 0.8)])
        assert result.should_alert is should_alert, label
        assert result.category.value == label
