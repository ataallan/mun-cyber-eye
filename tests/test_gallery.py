"""Objects gallery access, examples, and honest scan."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.factory import create_app


def _tiny_jpeg_bytes() -> bytes:
    img = np.zeros((32, 40, 3), dtype=np.uint8)
    img[:] = (30, 60, 90)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _write_tiny_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48))
    assert writer.isOpened()
    for i in range(4):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = (i * 20, 40, 80)
        writer.write(frame)
    writer.release()


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
            "ACTIVITY_DATA_ROOT": str(tmp_path / "activity"),
            "OBJECTS_DATA_ROOT": str(tmp_path / "objects"),
            "DANGEROUS_DATA_ROOT": str(tmp_path / "dangerous"),
            "CHECKPOINTS_DIR": str(tmp_path / "checkpoints"),
            "ACTIVE_CHECKPOINT_FILE": str(tmp_path / "active_checkpoint.json"),
            "ACTIVITY_CHECKPOINT": str(tmp_path / "checkpoints" / "activity.joblib"),
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "VISION_BACKEND": "mock",
            "ALERT_NOTIFY_ON_CREATE": False,
            "SEED_DEMO_CAMERAS": False,
            "MONITOR_AUTOSTART": False,
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client, username, password):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


def test_gallery_browse_is_for_signed_in_operators_not_anonymous(client):
    anon = client.get("/gallery")
    assert anon.status_code == 302
    assert "/login" in anon.headers["Location"]

    _login(client, "siteadmin", "test-pass-12")
    page = client.get("/gallery")
    body = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "Refrigerator" in body
    assert "Knife" in body
    assert "Unidentified striking object" in body
    assert "Needs labels" in body
    assert "YOLO" in body
    assert "Add examples" not in body

    denied = client.post(
        "/gallery/examples",
        data={"item": "dangerous:unidentified_improvised", "split": "train"},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "Only developer accounts can train models." in denied.get_data(as_text=True)


def test_operator_can_browse_but_not_train(client):
    _login(client, "reviewer", "reviewpass")
    body = client.get("/gallery").get_data(as_text=True)
    assert "Chair" in body
    assert "Add examples" not in body
    assert "Train object model" not in body


def test_developer_can_add_examples_for_unidentified_class(client, tmp_path):
    _login(client, "labdev", "dev-pass-99")
    page = client.get("/gallery")
    assert "Add examples" in page.get_data(as_text=True)
    saved = client.post(
        "/gallery/examples",
        data={
            "item": "dangerous:unidentified_improvised",
            "split": "train",
            "images": (io.BytesIO(_tiny_jpeg_bytes()), "hit.jpg"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert saved.status_code == 200
    dest = tmp_path / "dangerous" / "train" / "unidentified_improvised" / "hit.jpg"
    assert dest.is_file()
    body = saved.get_data(as_text=True)
    assert "Train 1" in body
    thumb = client.get(
        "/gallery/thumb/dangerous/unidentified_improvised/train/hit.jpg"
    )
    assert thumb.status_code == 200
    assert thumb.data[:2] == b"\xff\xd8"
    anon = client.application.test_client().get(
        "/gallery/thumb/dangerous/unidentified_improvised/train/hit.jpg"
    )
    assert anon.status_code == 302


def test_gallery_train_requires_confirmation_and_does_not_activate(client, app, monkeypatch):
    _login(client, "labdev", "dev-pass-99")
    before = app.config["ACTIVITY_CHECKPOINT"]
    called = {}

    def _fake_train(config, **kwargs):
        called["yes"] = True
        return Path(config["CHECKPOINTS_DIR"]) / "objects_custom.joblib", {}, {"categories": ["chair"]}

    monkeypatch.setattr("app.routes.training_ops.run_object_training", _fake_train)
    blocked = client.post(
        "/gallery/train",
        data={"output_name": "objects_custom.joblib"},
        follow_redirects=True,
    )
    assert "does not change live detection" in blocked.get_data(as_text=True)
    assert "yes" not in called
    assert app.config["ACTIVITY_CHECKPOINT"] == before

    allowed = client.post(
        "/gallery/train",
        data={"output_name": "objects_custom.joblib", "confirm_live_unchanged": "1"},
        follow_redirects=True,
    )
    text = allowed.get_data(as_text=True)
    assert called.get("yes") is True
    assert "Live detection is unchanged" in text
    assert app.config["ACTIVITY_CHECKPOINT"] == before


def test_scan_reports_unavailable_without_a_detector(client, tmp_path, monkeypatch):
    _login(client, "reviewer", "reviewpass")
    monkeypatch.setattr("vision.gallery.detect_gallery_labels", lambda image: None)
    clip = tmp_path / "clip.avi"
    _write_tiny_video(clip)
    page = client.post(
        "/gallery/scan",
        data={"clip": (clip.open("rb"), "clip.avi")},
        content_type="multipart/form-data",
    )
    body = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "unavailable" in body
    assert "Refrigerator" in body
    assert "mystery_blob" not in body


def test_scan_matches_gallery_and_drops_unknown_labels(client, tmp_path, monkeypatch):
    _login(client, "reviewer", "reviewpass")

    def _labels(_image):
        return [
            ("chair", 0.91),
            ("person", 0.88),
            ("knife", 0.7),
            ("mystery_blob", 0.95),
        ]

    monkeypatch.setattr("vision.gallery.detect_gallery_labels", _labels)
    clip = tmp_path / "clip.avi"
    _write_tiny_video(clip)
    page = client.post(
        "/gallery/scan",
        data={"clip": (clip.open("rb"), "clip.avi")},
        content_type="multipart/form-data",
    )
    body = page.get_data(as_text=True)
    assert "yolo" in body
    assert 'class="scan-match">Chair' in body
    assert 'class="scan-match">Knife' in body
    assert "mystery_blob" not in body
    assert 'class="scan-match">Refrigerator' not in body
