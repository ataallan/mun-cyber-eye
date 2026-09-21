"""Dangerous-object catalog and use-against-person intensity."""

from __future__ import annotations

import io

import numpy as np
import pytest

from app.factory import create_app
from risk.engine import ActivityCategory, RiskEngine
from vision.aggression import AggressionAssessment
from vision.dangerous_objects import (
    USE_BRANDISHED,
    USE_NONE,
    USE_STRIKE,
    USE_THREATENING,
    map_detector_label,
    validate_dangerous_id,
    weapon_should_soften_sport,
)
from vision.detector import Detection
from vision.weapon_assist import analyze_weapon_use


def test_catalog_maps_classes_and_harm():
    knife = map_detector_label("knife")
    assert knife is not None
    assert knife.weapon_class == "edged"
    assert knife.harm_potential == "high"
    assert knife.dangerous_object is True
    bat = map_detector_label("baseball bat")
    assert bat is not None
    assert bat.id == "baseball_bat"
    assert "baseball" in bat.sport_soften
    gun = map_detector_label("pistol")
    assert gun is not None
    assert gun.weapon_class == "firearm_like"
    bottle = map_detector_label("bottle")
    assert bottle is not None
    assert bottle.context_dependent is True
    with pytest.raises(ValueError, match="Unknown dangerous"):
        validate_dangerous_id("../etc")
    with pytest.raises(ValueError, match="Unknown dangerous"):
        validate_dangerous_id("refrigerator")


def test_intensity_tiers_none_brandished_threatening_strike():
    person = Detection("person", 0.9, (20, 40, 90, 280))
    other = Detection("person", 0.9, (240, 40, 320, 280))
    knife = Detection("knife", 0.8, (100, 180, 130, 210))
    none = analyze_weapon_use([person, knife])
    assert none.present is True
    assert none.weapon_id == "knife"
    assert none.harm_potential == "high"
    assert none.use_intensity_label in {USE_NONE, USE_BRANDISHED}
    assert none.aimed_at_person is False

    raised = analyze_weapon_use(
        [person, Detection("knife", 0.8, (40, 50, 70, 80)), Detection("raised_object", 0.7, (40, 50, 70, 80))]
    )
    assert raised.use_intensity_label == USE_BRANDISHED
    assert "brandished" in raised.cues
    assert "indicator only" in raised.note.lower()

    gun = Detection("gun", 0.8, (90, 160, 180, 180))
    aimed = analyze_weapon_use([person, other, gun])
    assert aimed.aimed_at_person is True
    assert aimed.use_intensity >= 0.9
    assert aimed.toward_person is True

    bat = Detection("baseball bat", 0.82, (200, 60, 230, 200))
    strike = analyze_weapon_use(
        [
            person,
            other,
            bat,
            Detection("raised_object", 0.7, (200, 40, 230, 90)),
            Detection("strike_motion", 0.75),
        ],
        aggression=AggressionAssessment(score=0.7, raised_arm=0.25),
    )
    assert strike.weapon_id == "baseball_bat"
    assert strike.use_intensity_label in {USE_STRIKE, USE_THREATENING}
    assert strike.use_intensity >= 0.68


def test_improvised_chair_and_bottle_need_use_cues():
    person = Detection("person", 0.9, (20, 40, 90, 280))
    idle = analyze_weapon_use(
        [person, Detection("chair", 0.88, (120, 160, 200, 280)), Detection("bottle", 0.8, (40, 200, 55, 240))]
    )
    assert idle.present is False
    assert idle.use_intensity_label == USE_NONE

    raised_chair = analyze_weapon_use(
        [
            person,
            Detection("person", 0.88, (240, 40, 320, 280)),
            Detection("chair", 0.85, (40, 40, 90, 100)),
            Detection("raised_object", 0.7, (40, 40, 90, 100)),
        ]
    )
    assert raised_chair.present is True
    assert raised_chair.weapon_id == "chair_as_weapon"
    assert raised_chair.use_intensity_label in {USE_BRANDISHED, USE_THREATENING, USE_STRIKE}


def test_sport_bat_low_intensity_softens_high_strike_alerts():
    extras = {
        "sport_context": "baseball",
        "sport_confidence": 0.88,
        "place_type": "sports_field",
        "place_confidence": 0.9,
        "place_source": "camera",
    }
    person = Detection("person", 0.9, (20, 40, 90, 280), extras=extras)
    other = Detection("person", 0.88, (320, 40, 400, 280), extras=extras)
    # Bat at the holder's hip — present, not raised, not toward the other player.
    bat = Detection("baseball bat", 0.8, (70, 190, 100, 250), extras=extras)
    low = analyze_weapon_use([person, other, bat])
    assert low.present is True
    assert low.use_intensity_label in {USE_NONE, USE_BRANDISHED}
    assert low.use_intensity < 0.65
    assert low.aimed_at_person is False
    extras_low = {
        **extras,
        "weapon": low.to_dict(),
        "weapon_id": low.weapon_id,
        "weapon_class": low.weapon_class,
        "harm_potential": low.harm_potential,
        "use_intensity_label": low.use_intensity_label,
        "weapon_use_intensity": low.use_intensity,
    }
    for det in (person, other, bat):
        det.extras = extras_low
    engine = RiskEngine()
    play = engine.assess([person, other, bat, Detection("game_or_play", 0.8, extras=extras_low)])
    assert weapon_should_soften_sport(
        weapon_id="baseball_bat",
        sport_context="baseball",
        sports_venue=True,
        sport_decent=True,
        use_intensity=low.use_intensity,
        use_intensity_label=low.use_intensity_label,
    )
    assert play.category == ActivityCategory.GAME_OR_PLAY
    assert play.should_alert is False
    assert play.weapon_id == "baseball_bat"

    street = {
        "place_type": "street",
        "place_confidence": 0.9,
        "place_source": "camera",
    }
    holder = Detection("person", 0.9, (20, 40, 90, 280), extras=street)
    target = Detection("person", 0.88, (240, 40, 320, 280), extras=street)
    swinging = Detection("baseball bat", 0.84, (200, 40, 240, 120), extras=street)
    raised = Detection("raised_object", 0.7, (200, 40, 240, 120), extras=street)
    strike = Detection("strike_motion", 0.8, extras=street)
    high = analyze_weapon_use(
        [holder, target, swinging, raised, strike],
        aggression=AggressionAssessment(score=0.72, raised_arm=0.3),
    )
    extras_high = {
        **street,
        "weapon": high.to_dict(),
        "weapon_id": high.weapon_id,
        "weapon_class": high.weapon_class,
        "harm_potential": high.harm_potential,
        "use_intensity_label": high.use_intensity_label,
        "weapon_use_intensity": high.use_intensity,
    }
    for det in (holder, target, swinging, raised, strike):
        det.extras = extras_high
    fightish = engine.assess([holder, target, swinging, raised, strike])
    assert fightish.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert fightish.should_alert is True
    assert fightish.use_intensity_label in {USE_STRIKE, USE_THREATENING}
    assert "indicator only" in fightish.rationale.lower()
    assert not weapon_should_soften_sport(
        weapon_id="baseball_bat",
        sport_context=None,
        sports_venue=False,
        sport_decent=False,
        use_intensity=high.use_intensity,
        use_intensity_label=high.use_intensity_label,
    )

    knife_ex = {
        "sport_context": "baseball",
        "sport_confidence": 0.88,
        "place_type": "sports_field",
        "place_confidence": 0.9,
        "place_source": "camera",
    }
    k_person = Detection("person", 0.9, (20, 40, 90, 280), extras=knife_ex)
    k_obj = Detection("knife", 0.82, (70, 190, 100, 220), extras=knife_ex)
    k_use = analyze_weapon_use([k_person, k_obj])
    knife_ex = {
        **knife_ex,
        "weapon": k_use.to_dict(),
        "weapon_id": k_use.weapon_id,
        "weapon_class": k_use.weapon_class,
        "harm_potential": k_use.harm_potential,
        "use_intensity_label": k_use.use_intensity_label,
        "weapon_use_intensity": k_use.use_intensity,
    }
    k_person.extras = knife_ex
    k_obj.extras = knife_ex
    knifed = engine.assess(
        [k_person, k_obj, Detection("game_or_play", 0.8, extras=knife_ex)]
    )
    assert knifed.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert knifed.should_alert is True
    assert knifed.weapon_id == "knife"
    assert not weapon_should_soften_sport(
        weapon_id="knife",
        sport_context="baseball",
        sports_venue=True,
        sport_decent=True,
        use_intensity=k_use.use_intensity,
        use_intensity_label=k_use.use_intensity_label,
    )


@pytest.fixture
def app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "operator",
            "ADMIN_PASSWORD": "changeme",
            "ADMIN_EMAIL": "operator@localhost",
            "ADMIN_SYNC_PASSWORD": True,
            "ADMIN_ROLE": "admin",
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "ACTIVITY_DATA_ROOT": str(tmp_path / "activity"),
            "OBJECTS_DATA_ROOT": str(tmp_path / "objects"),
            "DANGEROUS_DATA_ROOT": str(tmp_path / "dangerous"),
            "CHECKPOINTS_DIR": str(tmp_path / "checkpoints"),
            "ACTIVE_CHECKPOINT_FILE": str(tmp_path / "active_checkpoint.json"),
            "ACTIVITY_CHECKPOINT": str(tmp_path / "checkpoints" / "missing.joblib"),
            "ALERT_NOTIFY_ON_CREATE": False,
            "SEED_DEMO_CAMERAS": False,
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def _tiny_video(path, n_frames=8, fps=4):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (64, 48))
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = ((i * 20) % 255, 40, 80)
        writer.write(frame)
    writer.release()


def test_admin_extract_dangerous_and_weapon_activity(client, tmp_path):
    client.post(
        "/login",
        data={"username": "operator", "password": "changeme"},
        follow_redirects=True,
    )
    video = tmp_path / "clip.avi"
    _tiny_video(video)
    payload = video.read_bytes()
    page = client.get("/admin/train")
    html = page.get_data(as_text=True)
    assert "dangerous / weapon-like class" in html
    assert "potential_weapon_object" in html
    resp = client.post(
        "/admin/train/extract-video",
        data={
            "kind": "dangerous",
            "dangerous_id": "knife",
            "split": "train",
            "sample_fps": "3",
            "max_frames": "6",
            "video": (io.BytesIO(payload), "knife_cam.avi"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    dest = tmp_path / "dangerous" / "train" / "knife"
    assert list(dest.glob("*.jpg"))
    activity = client.post(
        "/admin/train/extract-video",
        data={
            "kind": "activity",
            "category": "potential_weapon_object",
            "split": "train",
            "sample_fps": "3",
            "max_frames": "4",
            "video": (io.BytesIO(payload), "weapon_class.avi"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert activity.status_code == 200
    assert list((tmp_path / "activity" / "train" / "potential_weapon_object").glob("*.jpg"))
