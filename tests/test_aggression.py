"""Body-aggression proxies and sport-vs-fight risk policy."""

import numpy as np

from risk.engine import ActivityCategory, RiskEngine, RiskLevel
from vision.aggression import analyze_aggression
from vision.detector import Detection


def _moving_blob_pair():
    """Localized motion + two close people — not a uniform scene cut."""
    prev = np.zeros((120, 160, 3), dtype=np.uint8)
    prev[:] = (30, 30, 30)
    prev[50:100, 10:40] = (20, 40, 180)
    prev[50:100, 120:150] = (20, 40, 200)
    cur = np.zeros((120, 160, 3), dtype=np.uint8)
    cur[:] = (30, 30, 30)
    cur[5:50, 45:85] = (20, 40, 180)  # upper-band motion (raised-arm proxy)
    cur[45:105, 55:100] = (20, 40, 200)
    return prev, cur


def test_aggression_cues_on_localized_motion():
    prev, cur = _moving_blob_pair()
    people = [
        Detection("person", 0.9, (20, 40, 50, 95)),
        Detection("person", 0.88, (70, 40, 115, 95)),
    ]
    result = analyze_aggression(cur, prev_bgr=prev, detections=people, prev_motion=0.05)
    assert result.uniform_scene_change is False
    assert result.score > 0.2
    assert set(result.cues) & {"aggressive_motion", "aggressive_pose", "close_proximity", "rapid_motion", "strike_motion"}


def test_uniform_scene_cut_is_not_aggression():
    a = np.zeros((80, 100, 3), dtype=np.uint8)
    a[:] = (10, 10, 10)
    b = np.zeros((80, 100, 3), dtype=np.uint8)
    b[:] = (200, 30, 30)
    result = analyze_aggression(b, prev_bgr=a)
    assert result.uniform_scene_change is True
    assert result.score < 0.35
    assert "aggressive_motion" not in result.cues


def _sport_extras(sport="basketball", conf=0.82, aggression=0.2, cues=None, place=None, place_conf=0.9, kit=0.0):
    extras = {
        "sport_context": sport,
        "sport_confidence": conf,
        "sport_display": "Basketball" if sport == "basketball" else sport,
        "aggression": {"score": aggression, "cues": cues or []},
        "face_aggression": {"status": "disabled", "note": "off"},
        "team_kit_similarity": kit,
        "jersey_like_colors": kit >= 0.55,
    }
    if place:
        extras["place_type"] = place
        extras["place_display"] = place.replace("_", " ")
        extras["place_confidence"] = place_conf
        extras["place_source"] = "test"
    return extras


def test_high_aggression_no_sport_leans_fight():
    engine = RiskEngine()
    result = engine.assess(
        [
            Detection(
                "ordinary",
                0.7,
                extras={
                    "aggression": {
                        "score": 0.78,
                        "cues": ["aggressive_motion", "strike_motion"],
                    }
                },
            ),
            Detection("aggressive_motion", 0.8),
            Detection("person", 0.9),
            Detection("person", 0.88),
        ]
    )
    assert result.category == ActivityCategory.POTENTIAL_FIGHT
    assert result.should_alert is True
    assert result.aggression_score >= 0.65


def test_high_aggression_with_sport_stays_play_no_alert():
    engine = RiskEngine()
    result = engine.assess(
        [
            Detection(
                "game_or_play",
                0.86,
                extras=_sport_extras(
                    aggression=0.8,
                    cues=["aggressive_motion", "rapid_motion"],
                ),
            ),
            Detection("aggressive_motion", 0.8),
            Detection("person", 0.9),
            Detection("person", 0.86),
        ]
    )
    assert result.category == ActivityCategory.GAME_OR_PLAY
    assert result.should_alert is False
    assert result.risk_level == RiskLevel.LOW
    assert "intense play" in result.rationale.lower()
    assert result.sport_context == "basketball"


def test_intense_sport_alerts_when_env_enabled():
    engine = RiskEngine(alert_on_intense_sport=True)
    result = engine.assess(
        [
            Detection(
                "game_or_play",
                0.84,
                extras=_sport_extras(
                    aggression=0.77,
                    cues=["aggressive_motion"],
                ),
            )
        ]
    )
    assert result.category == ActivityCategory.GAME_OR_PLAY
    assert result.should_alert is True
    assert "ALERT_ON_INTENSE_SPORT" in result.rationale


def test_low_aggression_with_sport_is_play():
    engine = RiskEngine()
    result = engine.assess(
        [
            Detection(
                "game_or_play",
                0.9,
                extras=_sport_extras(aggression=0.15, cues=[]),
            )
        ]
    )
    assert result.category == ActivityCategory.GAME_OR_PLAY
    assert result.should_alert is False
    assert "game or play" in result.rationale.lower()


def test_sport_context_softens_fight_heuristics():
    engine = RiskEngine()
    result = engine.assess(
        [
            Detection("person", 0.9, extras=_sport_extras(aggression=0.2)),
            Detection("person", 0.88, extras=_sport_extras(aggression=0.2)),
            Detection("close_proximity", 0.8, extras=_sport_extras(aggression=0.2)),
            Detection("rapid_motion", 0.82, extras=_sport_extras(aggression=0.2)),
        ]
    )
    assert result.category == ActivityCategory.GAME_OR_PLAY
    assert result.should_alert is False
    assert result.sport_context == "basketball"
    assert "softened" in result.rationale.lower() or "game or play" in result.rationale.lower()


def test_fight_heuristics_without_sport_still_alert():
    engine = RiskEngine()
    result = engine.assess(
        [
            Detection("person", 0.9),
            Detection("person", 0.88),
            Detection("close_proximity", 0.8),
            Detection("rapid_motion", 0.82),
            Detection("aggressive_motion", 0.7),
        ]
    )
    assert result.category == ActivityCategory.POTENTIAL_FIGHT
    assert result.should_alert is True


def test_aggressive_labels_are_fight_signals():
    assert "aggressive_motion" in RiskEngine.FIGHT_SIGNALS
    assert "aggressive_pose" in RiskEngine.FIGHT_SIGNALS


def test_sports_venue_plus_sport_softens_toward_play():
    engine = RiskEngine()
    result = engine.assess(
        [
            Detection(
                "person",
                0.9,
                extras=_sport_extras(
                    sport="basketball",
                    conf=0.48,
                    aggression=0.3,
                    place="basketball_court",
                ),
            ),
            Detection("person", 0.88, extras=_sport_extras(conf=0.48, place="basketball_court")),
            Detection("close_proximity", 0.8, extras=_sport_extras(conf=0.48, place="basketball_court")),
            Detection("rapid_motion", 0.82, extras=_sport_extras(conf=0.48, place="basketball_court")),
        ]
    )
    assert result.category == ActivityCategory.GAME_OR_PLAY
    assert result.should_alert is False
    assert result.place_type == "basketball_court"
    assert "sports venue" in result.rationale.lower() or "game or play" in result.rationale.lower()


def test_street_high_aggression_no_sport_leans_stronger_fight():
    engine = RiskEngine()
    result = engine.assess(
        [
            Detection(
                "ordinary",
                0.7,
                extras={
                    "place_type": "street",
                    "place_confidence": 0.9,
                    "place_source": "camera",
                    "aggression": {
                        "score": 0.78,
                        "cues": ["aggressive_motion", "strike_motion"],
                    },
                },
            ),
            Detection("person", 0.9),
            Detection("person", 0.88),
        ]
    )
    assert result.category == ActivityCategory.POTENTIAL_FIGHT
    assert result.should_alert is True
    assert result.risk_level == RiskLevel.HIGH
    assert "street" in result.rationale.lower()


def test_house_and_corridor_high_aggression_no_sport_alert():
    engine = RiskEngine()
    for place in ("house_interior", "corridor_hallway", "compound_courtyard"):
        result = engine.assess(
            [
                Detection(
                    "ordinary",
                    0.66,
                    extras={
                        "place_type": place,
                        "place_confidence": 0.88,
                        "aggression": {"score": 0.8, "cues": ["aggressive_motion"]},
                    },
                )
            ]
        )
        assert result.category == ActivityCategory.POTENTIAL_FIGHT
        assert result.should_alert is True
        assert result.risk_level == RiskLevel.HIGH


def test_street_plus_sport_stays_play_unless_extreme():
    engine = RiskEngine()
    play = engine.assess(
        [
            Detection(
                "game_or_play",
                0.84,
                extras=_sport_extras(
                    sport="soccer",
                    aggression=0.8,
                    cues=["aggressive_motion"],
                    place="street",
                ),
            )
        ]
    )
    assert play.category == ActivityCategory.GAME_OR_PLAY
    assert play.should_alert is False
    assert "street play" in play.rationale.lower()

    extreme = engine.assess(
        [
            Detection(
                "game_or_play",
                0.84,
                extras=_sport_extras(
                    sport="soccer",
                    aggression=0.91,
                    cues=["aggressive_motion", "strike_motion"],
                    place="street",
                ),
            )
        ]
    )
    assert extreme.category == ActivityCategory.POTENTIAL_FIGHT
    assert extreme.should_alert is True


def test_kit_similarity_boosts_play_confidence_absence_is_not_a_fight():
    engine = RiskEngine()
    boosted = engine.assess(
        [
            Detection(
                "game_or_play",
                0.7,
                extras=_sport_extras(conf=0.52, aggression=0.2, kit=0.8, place="sports_field"),
            )
        ]
    )
    assert boosted.category == ActivityCategory.GAME_OR_PLAY
    assert boosted.team_kit_similarity >= 0.55
    assert boosted.sport_confidence >= 0.52
    assert "identity" in boosted.rationale.lower() or "kit" in boosted.rationale.lower() or boosted.should_alert is False

    no_kit = engine.assess(
        [
            Detection(
                "game_or_play",
                0.86,
                extras=_sport_extras(aggression=0.15, kit=0.0, place="sports_field"),
            )
        ]
    )
    assert no_kit.category == ActivityCategory.GAME_OR_PLAY
    assert no_kit.should_alert is False


def test_weapon_and_fall_ignore_place_policy():
    engine = RiskEngine()
    weapon = engine.assess(
        [
            Detection("person", 0.9, extras={"place_type": "street", "place_confidence": 0.9}),
            Detection("knife", 0.8, extras={"place_type": "street", "place_confidence": 0.9}),
        ]
    )
    assert weapon.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert weapon.should_alert is True

    fall = engine.assess(
        [
            Detection(
                "person_down",
                0.82,
                extras={"place_type": "house_interior", "place_confidence": 0.9},
            )
        ]
    )
    assert fall.category == ActivityCategory.POTENTIAL_FALL
    assert fall.should_alert is True
