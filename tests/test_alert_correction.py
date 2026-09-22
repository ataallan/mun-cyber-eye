"""Developer label correction from alert review. Operators cannot train from it."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.factory import create_app
from vision.checkpoint_config import read_active_checkpoint


@pytest.fixture
def app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_EMAIL": "siteadmin@localhost",
            "ADMIN_SYNC_PASSWORD": True,
            "ADMIN_ROLE": "admin",
            "OPERATOR_USERNAME": "reviewer",
            "OPERATOR_PASSWORD": "reviewpass",
            "OPERATOR_EMAIL": "reviewer@localhost",
            "DEVELOPER_USERNAME": "labdev",
            "DEVELOPER_PASSWORD": "dev-pass-99",
            "DEVELOPER_EMAIL": "labdev@localhost",
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "CLIP_DIR": str(tmp_path / "clips"),
            "ARCHIVE_DIR": str(tmp_path / "archive"),
            "ARCHIVE_DB_PATH": str(tmp_path / "archive.db"),
            "ACTIVITY_DATA_ROOT": str(tmp_path / "activity"),
            "CHECKPOINTS_DIR": str(tmp_path / "checkpoints"),
            "ACTIVE_CHECKPOINT_FILE": str(tmp_path / "active_checkpoint.json"),
            "ACTIVITY_CHECKPOINT": str(tmp_path / "checkpoints" / "missing.joblib"),
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "VISION_BACKEND": "auto",
            "ALERT_NOTIFY_ON_CREATE": False,
            "SEED_DEMO_CAMERAS": False,
            "SAMPLE_FPS": 2,
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client, username="labdev", password="dev-pass-99"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


def _write_clip(path: Path, n_frames: int = 6, fps: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter MJPG unavailable in this environment")
    for i in range(n_frames):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[:] = (20 + i * 15, 60, 140)
        writer.write(img)
    writer.release()


def _alert(app, *, clip_path=None, camera_id="gate-1", category="potential_fight"):
    return app.extensions["alert_store"].create_alert(
        source_label="Authorized Camera — gate",
        category=category,
        risk_level="high",
        confidence=0.82,
        rationale="Possible confrontation on the gate camera.",
        frame_index=4,
        timestamp_sec=2.0,
        camera_id=camera_id,
        clip_path=clip_path,
    )


def _jpeg_count(root: Path) -> int:
    if not root.exists():
        return 0
    return len(list(root.rglob("*.jpg")))


def test_developer_sends_incident_clip_frames_without_activating(client, app, tmp_path):
    _login(client)
    clip = tmp_path / "clips" / "gate_incident.avi"
    _write_clip(clip)
    pointer = tmp_path / "active_checkpoint.json"
    pointer.write_text('{"path": "keep-me"}\n', encoding="utf-8")
    alert = _alert(app, clip_path=str(clip))

    page = client.get(f"/alerts/{alert.id}")
    body = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "Correct label" in body
    assert "Send to training" in body
    assert "Train and activate" in body
    assert "Incident clip" in body
    assert "No clip to learn from." not in body

    sent = client.post(
        f"/alerts/{alert.id}/correct-label",
        data={"intent": "send", "category": "dance", "sport_context": "", "place_type": ""},
        follow_redirects=True,
    )
    text = sent.get_data(as_text=True)
    assert sent.status_code == 200
    assert "Saved" in text
    assert "train/dance" in text
    assert "live checkpoint was not changed" in text
    assert 'value="dance" selected' in text
    dest = tmp_path / "activity" / "train" / "dance"
    frames = list(dest.glob("*.jpg"))
    assert frames
    assert all(frame.name.startswith(f"alert_{alert.id[:8]}_") for frame in frames)
    assert _jpeg_count(tmp_path / "activity" / "train" / "potential_fight") == 0
    assert pointer.read_text(encoding="utf-8") == '{"path": "keep-me"}\n'
    assert not list((tmp_path / "checkpoints").glob("*.joblib"))

    audit = app.extensions["alert_store"].audit_trail(alert.id)
    correction = next(row for row in audit if row["action"] == "correct_label")
    assert correction["actor"] == "labdev"
    assert "label=dance" in correction["note"]
    assert alert.id in correction["note"]
    system = app.extensions["alert_store"].list_system_audit()
    assert any(row["action"] == "correct_label" and row["actor"] == "labdev" for row in system)
    assert app.extensions["alert_store"].get(alert.id).status == "open"

    dismissed = client.post(
        f"/alerts/{alert.id}/action",
        data={"action": "dismiss", "note": "Reviewed after correction"},
        follow_redirects=True,
    )
    assert "dismissed" in dismissed.get_data(as_text=True)
    assert app.extensions["alert_store"].get(alert.id).status == "dismissed"
    assert list(dest.glob("*.jpg"))


def test_sport_and_place_frames_use_existing_folders(client, app, tmp_path):
    _login(client)
    clip = tmp_path / "clips" / "court.avi"
    _write_clip(clip)
    alert = _alert(app, clip_path=str(clip), category="game_or_play")

    sport = client.post(
        f"/alerts/{alert.id}/correct-label",
        data={
            "intent": "send",
            "category": "game_or_play",
            "sport_context": "basketball",
            "place_type": "",
        },
        follow_redirects=True,
    )
    assert "train/game_or_play__basketball" in sport.get_data(as_text=True)
    assert list((tmp_path / "activity" / "train" / "game_or_play__basketball").glob("*.jpg"))

    place_clip = tmp_path / "clips" / "street.avi"
    _write_clip(place_clip)
    fall = _alert(app, clip_path=str(place_clip), category="potential_fall")
    placed = client.post(
        f"/alerts/{fall.id}/correct-label",
        data={
            "intent": "send",
            "category": "potential_fall",
            "sport_context": "",
            "place_type": "street",
        },
        follow_redirects=True,
    )
    assert "train/potential_fall__scene__street" in placed.get_data(as_text=True)
    assert list(
        (tmp_path / "activity" / "train" / "potential_fall__scene__street").glob("*.jpg")
    )

    both = client.post(
        f"/alerts/{alert.id}/correct-label",
        data={
            "intent": "send",
            "category": "game_or_play",
            "sport_context": "basketball",
            "place_type": "street",
        },
        follow_redirects=True,
    )
    assert "Choose a sport context or a place." in both.get_data(as_text=True)


def test_missing_clip_and_unknown_class_are_honest(client, app, tmp_path):
    _login(client)
    bare = _alert(app, clip_path=None, camera_id="")
    page = client.get(f"/alerts/{bare.id}")
    assert "No clip to learn from." in page.get_data(as_text=True)
    assert "Send to training" not in page.get_data(as_text=True)

    missing = client.post(
        f"/alerts/{bare.id}/correct-label",
        data={"intent": "send", "category": "dance"},
        follow_redirects=True,
    )
    assert "No clip to learn from." in missing.get_data(as_text=True)
    assert _jpeg_count(tmp_path / "activity") == 0

    outside = tmp_path / "secret" / "outside.avi"
    _write_clip(outside)
    escaped = _alert(app, clip_path=str(outside), camera_id="")
    denied_path = client.post(
        f"/alerts/{escaped.id}/correct-label",
        data={"intent": "send", "category": "dance"},
        follow_redirects=True,
    )
    assert "No clip to learn from." in denied_path.get_data(as_text=True)
    assert _jpeg_count(tmp_path / "activity") == 0
    assert outside.is_file()

    clip = tmp_path / "clips" / "known.avi"
    _write_clip(clip)
    alert = _alert(app, clip_path=str(clip))
    unknown = client.post(
        f"/alerts/{alert.id}/correct-label",
        data={"intent": "send", "category": "not_a_class"},
        follow_redirects=True,
    )
    assert "Unknown activity category" in unknown.get_data(as_text=True)
    assert _jpeg_count(tmp_path / "activity") == 0
    assert not (tmp_path / "active_checkpoint.json").exists()


def test_archive_segment_is_used_when_the_incident_clip_is_missing(client, app, tmp_path):
    _login(client)
    segment = tmp_path / "archive" / "gate-1" / "segment.avi"
    _write_clip(segment)
    app.extensions["archive"].store.add_segment(
        camera_id="gate-1",
        relpath="gate-1/segment.avi",
        started_at="2000-01-01T00:00:00Z",
        ended_at="2099-01-01T00:00:00Z",
        nbytes=segment.stat().st_size,
    )
    alert = _alert(app, clip_path=str(tmp_path / "clips" / "missing.avi"), camera_id="gate-1")
    page = client.get(f"/alerts/{alert.id}")
    assert "Archive segment" in page.get_data(as_text=True)
    sent = client.post(
        f"/alerts/{alert.id}/correct-label",
        data={"intent": "send", "category": "ordinary"},
        follow_redirects=True,
    )
    assert "train/ordinary" in sent.get_data(as_text=True)
    assert list((tmp_path / "activity" / "train" / "ordinary").glob("*.jpg"))
    note = next(
        row["note"]
        for row in app.extensions["alert_store"].audit_trail(alert.id)
        if row["action"] == "correct_label"
    )
    assert "source=archive_segment" in note


def test_train_and_activate_requires_confirmation(client, app, tmp_path):
    _login(client)
    clip = tmp_path / "clips" / "confirm.avi"
    _write_clip(clip)
    alert = _alert(app, clip_path=str(clip))

    blocked = client.post(
        f"/alerts/{alert.id}/correct-label",
        data={"intent": "train_activate", "category": "dance"},
        follow_redirects=True,
    )
    assert "Confirm train and activate" in blocked.get_data(as_text=True)
    assert _jpeg_count(tmp_path / "activity") == 0
    assert not (tmp_path / "active_checkpoint.json").exists()
    assert not list((tmp_path / "checkpoints").glob("*.joblib"))

    activated = client.post(
        f"/alerts/{alert.id}/correct-label",
        data={
            "intent": "train_activate",
            "category": "dance",
            "confirm_train_activate": "1",
        },
        follow_redirects=True,
    )
    text = activated.get_data(as_text=True)
    assert activated.status_code == 200
    assert "activated" in text.lower()
    assert "train/dance" in text
    ckpt = tmp_path / "checkpoints" / "activity_custom.joblib"
    assert ckpt.is_file()
    record = read_active_checkpoint(tmp_path / "active_checkpoint.json")
    assert record is not None
    assert Path(record["path"]) == ckpt
    assert record["activated_by"] == "labdev"
    assert list((tmp_path / "activity" / "train" / "dance").glob("*.jpg"))
    actions = [row["action"] for row in app.extensions["alert_store"].list_system_audit()]
    assert "train_model" in actions
    assert "activate_checkpoint" in actions
    trail = [row["action"] for row in app.extensions["alert_store"].audit_trail(alert.id)]
    assert "correct_label" in trail
    assert "train_activate" in trail
    assert "Live checkpoint updated." in client.get(f"/alerts/{alert.id}").get_data(as_text=True)


def test_operator_and_site_admin_cannot_correct_labels(client, app, tmp_path):
    clip = tmp_path / "clips" / "hidden.avi"
    _write_clip(clip)
    alert = _alert(app, clip_path=str(clip))

    anonymous = client.post(
        f"/alerts/{alert.id}/correct-label",
        data={"intent": "send", "category": "dance"},
    )
    assert anonymous.status_code == 302
    assert "/login" in anonymous.headers["Location"]

    _login(client, "reviewer", "reviewpass")
    hidden = client.get(f"/alerts/{alert.id}")
    body = hidden.get_data(as_text=True)
    assert "Correct label" not in body
    assert "Send to training" not in body
    assert "Train and activate" not in body
    assert "Dismiss" in body
    denied = client.post(
        f"/alerts/{alert.id}/correct-label",
        data={"intent": "send", "category": "dance", "confirm_train_activate": "1"},
        follow_redirects=True,
    )
    assert "Only developer accounts can train models." in denied.get_data(as_text=True)
    assert _jpeg_count(tmp_path / "activity") == 0

    client.get("/logout", follow_redirects=True)
    _login(client, "siteadmin", "test-pass-12")
    admin_page = client.get(f"/alerts/{alert.id}")
    assert "Correct label" not in admin_page.get_data(as_text=True)
    admin_denied = client.post(
        f"/alerts/{alert.id}/correct-label",
        data={"intent": "train_activate", "category": "dance", "confirm_train_activate": "1"},
        follow_redirects=True,
    )
    assert "Only developer accounts can train models." in admin_denied.get_data(as_text=True)
    assert not (tmp_path / "active_checkpoint.json").exists()
    assert _jpeg_count(tmp_path / "activity") == 0
