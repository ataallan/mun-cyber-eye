"""Developer console training: auth gate, train, activate, upload."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.factory import create_app
from vision.activity import inspect_checkpoint
from vision.checkpoint_config import read_active_checkpoint
from vision.dataset import ACTIVITY_CATEGORIES
from vision.detector import create_adapter


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
            "CHECKPOINTS_DIR": str(tmp_path / "checkpoints"),
            "ACTIVE_CHECKPOINT_FILE": str(tmp_path / "active_checkpoint.json"),
            "ACTIVITY_CHECKPOINT": str(tmp_path / "checkpoints" / "missing.joblib"),
            "VISION_BACKEND": "auto",
            "ALERT_NOTIFY_ON_CREATE": False,
            "SEED_DEMO_CAMERAS": False,
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


def _tiny_jpeg_bytes(seed: int = 1) -> bytes:
    img = np.zeros((40, 48, 3), dtype=np.uint8)
    img[:] = (20 + seed, 40, 80)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_train_page_requires_login(client):
    resp = client.get("/admin/train")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_operator_forbidden_from_train_actions(client, tmp_path):
    _login(client, "reviewer", "reviewpass")
    page = client.get("/admin/train", follow_redirects=True)
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert ">Train models</a>" not in body
    assert "Only developer accounts can train models." in body
    assert "Train from labeled data" not in body
    assert "Extract frames from video" not in body

    denied = client.post(
        "/admin/train/run",
        data={
            "model_type": "forest",
            "output_name": "activity_custom.joblib",
            "generate_demo": "1",
        },
        follow_redirects=True,
    )
    assert denied.status_code == 200
    assert "Only developer accounts can train models." in denied.get_data(as_text=True)
    assert not list((tmp_path / "checkpoints").glob("*.joblib"))

    activate = client.post(
        "/admin/train/activate",
        data={"checkpoint_path": "activity_custom.joblib"},
        follow_redirects=True,
    )
    assert "Only developer accounts can train models." in activate.get_data(as_text=True)


def test_site_admin_cannot_train(client, tmp_path):
    _login(client, "siteadmin", "test-pass-12")
    dash = client.get("/")
    assert ">Train models</a>" not in dash.get_data(as_text=True)
    hidden = client.get("/admin/train", follow_redirects=True)
    assert "Only developer accounts can train models." in hidden.get_data(as_text=True)
    denied = client.post(
        "/admin/train/run",
        data={
            "model_type": "forest",
            "output_name": "activity_custom.joblib",
            "generate_demo": "1",
        },
        follow_redirects=True,
    )
    assert "Only developer accounts can train models." in denied.get_data(as_text=True)
    assert not list((tmp_path / "checkpoints").glob("*.joblib"))


def test_developer_nav_shows_train_models(client):
    _login(client)
    dash = client.get("/")
    assert ">Train models</a>" in dash.get_data(as_text=True)
    alias = client.get("/train")
    assert alias.status_code == 302
    assert "/admin/train" in alias.headers["Location"]


def test_developer_can_train_activate_and_upload(client, app, tmp_path):
    _login(client)
    resp = client.post(
        "/admin/train/run",
        data={
            "model_type": "forest",
            "output_name": "activity_custom.joblib",
            "generate_demo": "1",
            "n_train": "6",
            "n_val": "3",
            "n_test": "3",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Training finished" in body
    assert "accuracy:" in body
    ckpt = tmp_path / "checkpoints" / "activity_custom.joblib"
    assert ckpt.is_file()
    info = inspect_checkpoint(ckpt)
    assert info["loads"] is True
    for cat in ACTIVITY_CATEGORIES:
        assert cat in info["categories"]
    assert "game_or_play" in info["categories"]
    assert "dance" in info["categories"]
    assert "potential_fight" in info["categories"]

    activated = client.post(
        "/admin/train/activate",
        data={"checkpoint_path": str(ckpt)},
        follow_redirects=True,
    )
    assert activated.status_code == 200
    assert "Active checkpoint is now" in activated.get_data(as_text=True)
    record = read_active_checkpoint(tmp_path / "active_checkpoint.json")
    assert record is not None
    assert Path(record["path"]) == ckpt
    assert record["activated_by"] == "labdev"
    assert app.config["ACTIVITY_CHECKPOINT"] == str(ckpt)

    upload = client.post(
        "/admin/train/upload",
        data={
            "category": "potential_fight",
            "split": "train",
            "images": (io.BytesIO(_tiny_jpeg_bytes()), "site_fight.jpg"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert upload.status_code == 200
    assert "Saved 1 labeled frame" in upload.get_data(as_text=True)
    dest = tmp_path / "activity" / "train" / "potential_fight" / "site_fight.jpg"
    assert dest.is_file()

    actions = [row["action"] for row in app.extensions["alert_store"].list_system_audit()]
    assert "train_model" in actions
    assert "activate_checkpoint" in actions
    assert "upload_labels" in actions


def test_upload_rejects_unknown_category_and_zip_traversal(client, tmp_path):
    _login(client)
    bad = client.post(
        "/admin/train/upload",
        data={
            "category": "../etc",
            "split": "train",
            "images": (io.BytesIO(_tiny_jpeg_bytes()), "x.jpg"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "Unknown activity category" in bad.get_data(as_text=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../escape.jpg", _tiny_jpeg_bytes())
    buf.seek(0)
    zip_resp = client.post(
        "/admin/train/upload",
        data={
            "category": "ordinary",
            "split": "train",
            "zipfile": (buf, "frames.zip"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "path traversal" in zip_resp.get_data(as_text=True).lower()
    assert not (tmp_path / "escape.jpg").exists()


def test_create_adapter_reads_active_checkpoint_json(tmp_path, monkeypatch, app):
    _login_client = app.test_client()
    _login(_login_client)
    _login_client.post(
        "/admin/train/run",
        data={
            "model_type": "forest",
            "output_name": "activity_custom.joblib",
            "generate_demo": "1",
            "n_train": "6",
            "n_val": "2",
            "n_test": "2",
        },
    )
    ckpt = tmp_path / "checkpoints" / "activity_custom.joblib"
    _login_client.post("/admin/train/activate", data={"checkpoint_path": str(ckpt)})

    monkeypatch.delenv("ACTIVITY_CHECKPOINT", raising=False)
    monkeypatch.setenv("ACTIVE_CHECKPOINT_FILE", str(tmp_path / "active_checkpoint.json"))
    monkeypatch.setenv("VISION_BACKEND", "auto")
    adapter = create_adapter()
    assert adapter.name == "activity"
    assert Path(adapter.checkpoint_path) == ckpt


def test_developer_zip_upload_uses_split_category_layout(client, tmp_path):
    _login(client)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("train/ordinary/lab_frame.jpg", _tiny_jpeg_bytes(2))
    buf.seek(0)
    resp = client.post(
        "/admin/train/upload",
        data={
            "category": "potential_fall",
            "split": "train",
            "zipfile": (buf, "frames.zip"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert "Saved 1 labeled frame" in resp.get_data(as_text=True)
    dest = tmp_path / "activity" / "train" / "ordinary" / "lab_frame.jpg"
    assert dest.is_file()


def test_developer_upload_preserves_sport_context_folder(client, tmp_path):
    _login(client)
    resp = client.post(
        "/admin/train/upload",
        data={
            "category": "game_or_play",
            "sport_context": "basketball",
            "split": "train",
            "images": (io.BytesIO(_tiny_jpeg_bytes(3)), "hoops.jpg"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    dest = tmp_path / "activity" / "train" / "game_or_play__basketball" / "hoops.jpg"
    assert dest.is_file()
    body = resp.get_data(as_text=True)
    assert "basketball" in body.lower()


def test_developer_upload_scene_place_folder(client, tmp_path):
    _login(client)
    resp = client.post(
        "/admin/train/upload",
        data={
            "category": "ordinary",
            "place_type": "street",
            "split": "train",
            "images": (io.BytesIO(_tiny_jpeg_bytes(6)), "block.jpg"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    dest = tmp_path / "activity" / "train" / "scene__street" / "block.jpg"
    assert dest.is_file()
    page = client.get("/admin/train")
    body = page.get_data(as_text=True)
    assert "scene__" in body or "Place type" in body or "street" in body.lower()


def test_developer_zip_accepts_scene_folder(client, tmp_path):
    _login(client)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("train/scene__corridor_hallway/hall.jpg", _tiny_jpeg_bytes(7))
    buf.seek(0)
    resp = client.post(
        "/admin/train/upload",
        data={"zipfile": (buf, "scene.zip")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert "Saved 1 labeled frame" in resp.get_data(as_text=True)
    assert (
        tmp_path / "activity" / "train" / "scene__corridor_hallway" / "hall.jpg"
    ).is_file()


def test_developer_upload_custom_place_folder(client, app, tmp_path):
    _login(client)
    page = client.get("/admin/train")
    assert "Other / custom" in page.get_data(as_text=True)
    resp = client.post(
        "/admin/train/upload",
        data={
            "category": "ordinary",
            "place_type": "__custom__",
            "place_type_custom": "rooftop café",
            "split": "train",
            "images": (io.BytesIO(_tiny_jpeg_bytes(8)), "roof.jpg"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    dest = tmp_path / "activity" / "train" / "scene__rooftop_cafe" / "roof.jpg"
    assert dest.is_file()
    choices = {p.id for p in app.extensions["camera_store"].list_place_choices()}
    assert "rooftop_cafe" in choices
    again = client.get("/admin/train")
    assert "rooftop_cafe" in again.get_data(as_text=True)


def test_developer_zip_accepts_catalog_sport_folder(client, tmp_path):
    _login(client)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("train/basketball/court.jpg", _tiny_jpeg_bytes(4))
        zf.writestr("train/game_or_play__soccer/pitch.jpg", _tiny_jpeg_bytes(5))
    buf.seek(0)
    resp = client.post(
        "/admin/train/upload",
        data={"zipfile": (buf, "sports.zip")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert "Saved 2 labeled frame" in resp.get_data(as_text=True)
    assert (tmp_path / "activity" / "train" / "game_or_play__basketball" / "court.jpg").is_file()
    assert (tmp_path / "activity" / "train" / "game_or_play__soccer" / "pitch.jpg").is_file()


def test_overwrite_demo_requires_confirmation(client, app, tmp_path):
    _login(client)
    dest = tmp_path / "checkpoints" / "activity_demo.joblib"
    dest.write_bytes(b"placeholder")
    resp = client.post(
        "/admin/train/run",
        data={
            "model_type": "forest",
            "output_name": "activity_demo.joblib",
            "generate_demo": "1",
            "n_train": "4",
            "n_val": "2",
            "n_test": "2",
        },
        follow_redirects=True,
    )
    assert "requires confirmation" in resp.get_data(as_text=True)
    assert dest.read_bytes() == b"placeholder"


def test_first_register_site_admin_cannot_train(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "",
            "ADMIN_PASSWORD": "",
            "DEVELOPER_USERNAME": "",
            "DEVELOPER_PASSWORD": "",
            "ADMIN_SYNC_PASSWORD": False,
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "ACTIVITY_DATA_ROOT": str(tmp_path / "activity"),
            "CHECKPOINTS_DIR": str(tmp_path / "checkpoints"),
            "ACTIVE_CHECKPOINT_FILE": str(tmp_path / "active_checkpoint.json"),
            "ACTIVITY_CHECKPOINT": str(tmp_path / "checkpoints" / "missing.joblib"),
            "ALERT_NOTIFY_ON_CREATE": False,
            "SEED_DEMO_CAMERAS": False,
        }
    )
    client = application.test_client()
    created = client.post(
        "/register",
        data={
            "username": "founder",
            "email": "founder@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
        },
        follow_redirects=True,
    )
    assert created.status_code == 200
    founder = application.extensions["user_store"].get_by_username("founder")
    assert founder is not None
    assert founder.role == "admin"
    client.post(
        "/login",
        data={"username": "founder", "password": "secret123"},
        follow_redirects=True,
    )
    dash = client.get("/")
    assert ">Train models</a>" not in dash.get_data(as_text=True)
    denied = client.post(
        "/admin/train/run",
        data={
            "model_type": "forest",
            "output_name": "activity_custom.joblib",
            "generate_demo": "1",
        },
        follow_redirects=True,
    )
    assert "Only developer accounts can train models." in denied.get_data(as_text=True)
    assert not list((tmp_path / "checkpoints").glob("*.joblib"))
    upload = client.post(
        "/admin/train/upload",
        data={"category": "ordinary", "split": "train"},
        follow_redirects=True,
    )
    assert "Only developer accounts can train models." in upload.get_data(as_text=True)
    extract = client.post(
        "/admin/train/extract-video",
        data={"kind": "activity", "category": "ordinary"},
        follow_redirects=True,
    )
    assert "Only developer accounts can train models." in extract.get_data(as_text=True)

