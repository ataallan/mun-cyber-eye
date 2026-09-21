"""Fall manner, gunshot audio honesty, aimed firearm, thrown-object assists."""

from __future__ import annotations

import numpy as np
import pytest

from alerts.schema import category_display_name, recommended_human_action
from alerts.store import AlertStore
from app.factory import create_app
from ingest.sampler import SampledFrame
from pipeline import CyberEyePipeline
from risk.engine import ActivityCategory, RiskEngine, RiskLevel
from vision.aggression import AggressionAssessment
from vision.detector import Detection, VisionAdapter
from vision.fall_assist import (
    ACCIDENTAL_FALL,
    SUDDEN_COLLAPSE,
    UNKNOWN_FALL,
    analyze_fall_manner,
)
from vision.gunshot_assist import analyze_gunshot_audio, analyze_gunshot_proxy
from vision.throw_assist import (
    THROW_LABEL,
    USE_THROWN,
    ObjectTrack,
    ThrowAssessment,
    analyze_throw,
    throw_should_soften,
    tracks_from_detections,
)
from vision.weapon_assist import analyze_weapon_use


def test_potential_gunshot_is_risk_enum_not_sklearn_class():
    from vision.dataset import ACTIVITY_CATEGORIES

    assert ActivityCategory.POTENTIAL_GUNSHOT.value == "potential_gunshot"
    assert "potential_gunshot" not in ACTIVITY_CATEGORIES
    sklearn_cats = {
        c.value for c in ActivityCategory if c != ActivityCategory.POTENTIAL_GUNSHOT
    }
    assert set(ACTIVITY_CATEGORIES) == sklearn_cats
    assert category_display_name("potential_gunshot") == "potential gunshot (video proxy)"
    action = recommended_human_action("potential_gunshot")
    assert "Advisory only" in action
    assert "not a confirmed gunshot" in action.lower() or "not ballistic" in action.lower()


def test_fall_manner_split_still_alerts():
    engine = RiskEngine()
    standing = [(40.0, 20.0, 90.0, 200.0)]
    sudden = analyze_fall_manner(
        [
            Detection("person", 0.9, (40, 160, 200, 230)),
            Detection("person_down", 0.84, (40, 160, 200, 230)),
        ],
        aggression=AggressionAssessment(
            score=0.72, sudden_acceleration=0.6, motion_intensity=0.5
        ),
        prev_person_boxes=standing,
    )
    assert sudden.manner in {SUDDEN_COLLAPSE, "unprecedented_fall"}
    assert sudden.applicable is True

    accidental = analyze_fall_manner(
        [
            Detection("person", 0.88, (20, 180, 220, 240)),
            Detection("horizontal_pose", 0.8, (20, 180, 220, 240)),
        ],
        aggression=AggressionAssessment(
            score=0.2, sudden_acceleration=0.05, motion_intensity=0.08
        ),
        prev_person_boxes=None,
    )
    assert accidental.manner == ACCIDENTAL_FALL

    unknown = analyze_fall_manner(
        [Detection("potential_fall", 0.7, (10, 10, 80, 90))],
        aggression=AggressionAssessment(score=0.4, sudden_acceleration=0.2),
    )
    assert unknown.manner in {UNKNOWN_FALL, ACCIDENTAL_FALL, SUDDEN_COLLAPSE}

    for manner, extras in (
        (sudden.manner, sudden.to_dict()),
        (accidental.manner, accidental.to_dict()),
        (unknown.manner, unknown.to_dict()),
    ):
        result = engine.assess(
            [
                Detection(
                    "person_down",
                    0.81,
                    extras={"fall": extras, "fall_manner": manner},
                ),
            ]
        )
        assert result.category == ActivityCategory.POTENTIAL_FALL
        assert result.should_alert is True
        assert result.fall_manner == manner


def test_gunshot_audio_never_invented(monkeypatch, tmp_path):
    monkeypatch.delenv("ENABLE_GUNSHOT_AUDIO", raising=False)
    disabled = analyze_gunshot_audio("/tmp/missing.wav")
    assert disabled["status"] == "disabled"
    assert disabled["confidence"] == 0.0
    assert "invent" in disabled["note"].lower() or "off" in disabled["note"].lower()

    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF")
    still_off = analyze_gunshot_audio(wav)
    assert still_off["status"] == "disabled"
    assert still_off["confidence"] == 0.0

    monkeypatch.setenv("ENABLE_GUNSHOT_AUDIO", "1")
    missing = analyze_gunshot_audio(None)
    assert missing["status"] == "unavailable"
    assert missing["confidence"] == 0.0
    present = analyze_gunshot_audio(wav)
    assert present["status"] == "unavailable"
    assert present["confidence"] == 0.0
    assert "invent" in present["note"].lower()

    blank = np.zeros((48, 64, 3), dtype=np.uint8)
    proxy = analyze_gunshot_proxy(blank, [Detection("person", 0.9)])
    assert proxy.video_proxy is False
    assert proxy.audio_status in {"disabled", "unavailable"}
    assert proxy.audio_confidence == 0.0


def test_aimed_firearm_two_people_vs_single_person_brandish():
    holder = Detection("person", 0.93, (10, 40, 80, 300))
    other = Detection("person", 0.91, (240, 40, 320, 300))
    gun = Detection("gun", 0.8, (80, 160, 170, 180))
    two = analyze_weapon_use([holder, other, gun])
    assert two.aimed_at_person is True
    assert two.use_tier == "aimed_at_person"
    assert "firearm_aimed_at_person" in two.cues

    one = analyze_weapon_use([holder, gun])
    assert one.aimed_at_person is False
    assert one.present is True
    assert "brandish_only" in one.cues

    engine = RiskEngine()
    aimed = engine.assess(
        [
            holder,
            other,
            gun,
            Detection(
                "firearm_aimed_at_person",
                0.78,
                extras={"weapon": two.to_dict(), "aimed_at_person": True},
            ),
        ]
    )
    assert aimed.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert aimed.should_alert is True
    assert aimed.risk_level == RiskLevel.HIGH
    assert aimed.aimed_at_person is True
    assert "pointed toward a person" in aimed.rationale.lower()

    brandish = engine.assess([holder, gun])
    assert brandish.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert brandish.aimed_at_person is False


def test_sport_context_does_not_clear_aimed_firearm():
    extras = {
        "sport_context": "basketball",
        "sport_confidence": 0.9,
        "place_type": "basketball_court",
        "place_confidence": 0.9,
        "aimed_at_person": True,
        "weapon": {
            "aimed_at_person": True,
            "use_tier": "aimed_at_person",
            "use_intensity": 0.92,
        },
        "aggression": {"score": 0.2, "cues": []},
    }
    engine = RiskEngine()
    result = engine.assess(
        [
            Detection("game_or_play", 0.86, extras=extras),
            Detection("person", 0.9, extras=extras),
            Detection("person", 0.88, extras=extras),
            Detection("gun", 0.8, extras=extras),
            Detection("firearm_aimed_at_person", 0.8, extras=extras),
        ]
    )
    assert result.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert result.should_alert is True
    assert result.aimed_at_person is True
    assert "sport context does not suppress" in result.rationale.lower()


def _throw_pair(label: str, extras: dict | None = None):
    holder = Detection("person", 0.92, (20, 40, 90, 280))
    target = Detection("person", 0.9, (260, 40, 330, 280))
    prev = ObjectTrack(label=label, bbox=(70, 150, 95, 185), center=(82.5, 167.5))
    current = Detection(label, 0.8, (210, 150, 235, 185), extras=extras or {})
    return [holder, target, current], [prev]


def test_throw_toward_person_alerts_street_bottle_not_sport_ball():
    engine = RiskEngine()
    bottle_dets, prev = _throw_pair("bottle")
    assessment = analyze_throw(bottle_dets, prev_tracks=prev)
    assert assessment.thrown_at_person is True
    assert assessment.harmful is True
    assert assessment.use_tier == USE_THROWN
    street = {
        "place_type": "street",
        "place_confidence": 0.88,
        "throw": assessment.to_dict(),
        "thrown_at_person": True,
    }
    bottle_alert = engine.assess(
        [
            Detection("person", 0.9, extras=street),
            Detection("person", 0.88, extras=street),
            Detection("bottle", 0.8, extras=street),
            Detection(THROW_LABEL, assessment.confidence, extras=street),
        ]
    )
    assert bottle_alert.should_alert is True
    assert bottle_alert.thrown_at_person is True
    assert bottle_alert.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert "thrown toward a person" in bottle_alert.rationale.lower()
    assert "not proof of assault" in bottle_alert.rationale.lower()
    assert THROW_LABEL in bottle_alert.contributing_labels

    ball_dets, ball_prev = _throw_pair("sports ball")
    ball = analyze_throw(ball_dets, prev_tracks=ball_prev)
    assert ball.thrown_at_person is True
    assert ball.sport_projectile is True
    play = {
        "sport_context": "basketball",
        "sport_confidence": 0.85,
        "place_type": "basketball_court",
        "place_confidence": 0.9,
        "throw": ball.to_dict(),
        "thrown_at_person": True,
    }
    pass_result = engine.assess(
        [
            Detection("game_or_play", 0.84, extras=play),
            Detection("person", 0.9, extras=play),
            Detection("person", 0.88, extras=play),
            Detection("sports ball", 0.8, extras=play),
            Detection(THROW_LABEL, ball.confidence, extras=play),
        ]
    )
    assert pass_result.category == ActivityCategory.GAME_OR_PLAY
    assert pass_result.should_alert is False
    assert pass_result.thrown_at_person is True

    assert (
        throw_should_soften(
            ThrowAssessment(thrown_at_person=True, harmful=True, sport_projectile=False),
            confrontation_setting=True,
        )
        is False
    )
    assert (
        throw_should_soften(
            ThrowAssessment(
                thrown_at_person=True, harmful=False, sport_projectile=True
            ),
            sport_context="soccer",
            sports_venue=True,
        )
        is True
    )


def test_throw_distinct_from_aimed_firearm_and_static_brandish():
    gun = Detection("gun", 0.8, (80, 160, 170, 180))
    holder = Detection("person", 0.9, (10, 40, 80, 300))
    other = Detection("person", 0.9, (240, 40, 320, 300))
    thrown = analyze_throw(
        [holder, other, gun],
        prev_tracks=[ObjectTrack("gun", (20, 160, 50, 180), (35, 170))],
    )
    assert thrown.thrown_at_person is False  # firearms are skipped

    static_bottle = analyze_throw(
        [
            holder,
            other,
            Detection("bottle", 0.8, (70, 150, 95, 185)),
        ],
        prev_tracks=[ObjectTrack("bottle", (68, 150, 93, 185), (80.5, 167.5))],
    )
    assert static_bottle.thrown_at_person is False  # no fast translation

    engine = RiskEngine()
    both = {
        "aimed_at_person": True,
        "thrown_at_person": True,
        "throw": {
            "thrown_at_person": True,
            "harmful": True,
            "object_label": "bottle",
            "use_tier": USE_THROWN,
            "confidence": 0.6,
        },
        "weapon": {"aimed_at_person": True, "use_tier": "aimed_at_person"},
    }
    result = engine.assess(
        [
            Detection("person", 0.9, extras=both),
            Detection("person", 0.88, extras=both),
            Detection("gun", 0.7, extras=both),
            Detection("bottle", 0.7, extras=both),
            Detection("firearm_aimed_at_person", 0.8, extras=both),
            Detection(THROW_LABEL, 0.6, extras=both),
        ]
    )
    assert result.aimed_at_person is True
    assert result.thrown_at_person is True
    assert result.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert "pointed toward a person" in result.rationale.lower()


class _ThrowAdapter(VisionAdapter):
    name = "throw-test"

    def detect(self, image_bgr, frame_index: int = 0):
        if frame_index == 0:
            return [
                Detection("person", 0.92, (20, 40, 90, 280)),
                Detection("person", 0.9, (260, 40, 330, 280)),
                Detection("bottle", 0.8, (70, 150, 95, 185)),
            ]
        return [
            Detection("person", 0.92, (20, 40, 90, 280)),
            Detection("person", 0.9, (260, 40, 330, 280)),
            Detection("bottle", 0.8, (210, 150, 235, 185)),
        ]


def test_pipeline_throw_metadata_and_last_run_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vision.object_inventory.try_load_yolo", lambda model_name="yolov8n.pt": None
    )
    store = AlertStore(tmp_path / "alerts.db")
    pipe = CyberEyePipeline(
        store=store,
        adapter=_ThrowAdapter(),
        snapshot_dir=tmp_path / "snaps",
        object_mode="auto",
        camera_place_type="street",
    )
    frames = [
        SampledFrame(
            index=i,
            timestamp_sec=i / 2.0,
            image_bgr=np.zeros((120, 160, 3), dtype=np.uint8),
            source_label="authorized throw clip",
        )
        for i in range(2)
    ]
    result = pipe.run_frames(frames, source_label="authorized throw clip")
    thrown_assess = [a for a in result.assessments if a.thrown_at_person]
    assert thrown_assess
    assert any(a.should_alert for a in thrown_assess)
    alerts = result.alerts_created
    assert alerts
    meta = alerts[-1].to_dict().get("metadata") or {}
    assert meta.get("throw", {}).get("cue") == "object_thrown_at_person"
    assert meta["throw"]["use_tier"] == "thrown_projectile"
    assert "thrown toward a person" in (alerts[-1].rationale or "").lower()
    assert meta.get("throw", {}).get("thrown_at_person") is True


@pytest.fixture
def app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "ADMIN_USERNAME": "operator",
            "ADMIN_PASSWORD": "changeme",
            "ADMIN_ROLE": "admin",
            "ALERT_DB_PATH": str(tmp_path / "app.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snaps"),
            "VISION_BACKEND": "mock",
            "ALERT_NOTIFY_ON_CREATE": False,
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def test_last_run_and_alert_detail_show_safety_cues(app, client):
    client.post(
        "/login",
        data={"username": "operator", "password": "changeme"},
        follow_redirects=True,
    )
    run = client.post("/run", data={"mode": "synthetic"}, follow_redirects=True)
    assert run.status_code == 200
    body = run.get_data(as_text=True)
    assert "object_thrown_at_person" in body
    assert "firearm_aimed_at_person" in body
    assert "thrown_projectile" in body
    assert "Fall manner" in body
    assert "Gunshot video proxy" in body
    assert "Safety cues" in body

    store = app.extensions["alert_store"]
    thrown = next(
        (
            a
            for a in store.list_alerts()
            if (a.to_dict().get("metadata") or {})
            .get("throw", {})
            .get("thrown_at_person")
            or "thrown toward a person" in (a.rationale or "").lower()
        ),
        None,
    )
    aimed = next(
        (
            a
            for a in store.list_alerts()
            if (a.to_dict().get("metadata") or {})
            .get("weapon", {})
            .get("aimed_at_person")
            or "pointed toward a person" in (a.rationale or "").lower()
        ),
        None,
    )
    fall = next((a for a in store.list_alerts() if a.category == "potential_fall"), None)
    assert thrown is not None
    detail = client.get(f"/alerts/{thrown.id}")
    html = detail.get_data(as_text=True)
    assert "object_thrown_at_person" in html
    assert "thrown_projectile" in html
    assert "not proof of assault" in html.lower()
    if aimed is not None:
        aimed_html = client.get(f"/alerts/{aimed.id}").get_data(as_text=True)
        assert "firearm_aimed_at_person" in aimed_html
    if fall is not None:
        fall_html = client.get(f"/alerts/{fall.id}").get_data(as_text=True)
        assert "fall" in fall_html.lower()


def test_health_reports_gunshot_audio_off(client):
    data = client.get("/health").get_json()
    assert data["gunshot_audio_enabled"] is False
    assert data["objects_catalog_size"] >= 40


def test_tracks_from_detections_skip_people_and_firearms():
    tracks = tracks_from_detections(
        [
            Detection("person", 0.9, (0, 0, 10, 40)),
            Detection("gun", 0.8, (4, 4, 20, 10)),
            Detection("bottle", 0.7, (8, 8, 18, 28)),
        ]
    )
    assert [t.label for t in tracks] == ["bottle"]
