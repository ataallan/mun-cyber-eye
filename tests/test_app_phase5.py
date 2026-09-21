"""Flask console: camera registry, run-from-camera, webcam gate, health."""

import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from app.factory import create_app
from vision.detector import Detection, VisionAdapter


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
            "ALLOW_WEBCAM": False,
            "RTSP_CONNECT_TIMEOUT_SEC": 1,
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client, username="operator", password="changeme"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


def test_health_reports_phase_5_and_cameras(client):
    data = client.get("/health").get_json()
    assert data["phase"] == 5
    assert data["allow_webcam"] is False
    assert data["cameras"]["total"] >= 2
    assert data["cameras"]["enabled"] >= 1
    assert data["place_catalog_size"] >= 20
    assert data["objects_catalog_size"] >= 40
    assert data["dangerous_objects_catalog_size"] >= 10
    assert data["gunshot_audio_enabled"] is False


def test_cameras_requires_auth(client):
    resp = client.get("/cameras")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_camera_crud_masks_rtsp_password(app, client):
    _login(client)
    resp = client.post(
        "/cameras/new",
        data={
            "name": "Gate RTSP",
            "location_label": "North gate",
            "source_type": "rtsp",
            "uri": "rtsp://admin:plain-secret@nvr.example/stream1",
            "sample_fps": "2",
            "enabled": "1",
            "notes": "Authorized NVR only",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Gate RTSP" in body
    assert "North gate" in body
    assert "plain-secret" not in body
    assert "****" in body
    stored = [
        c for c in app.extensions["camera_store"].list_cameras() if c.name == "Gate RTSP"
    ]
    assert stored
    assert "plain-secret" in stored[0].uri


def test_disable_seeded_rtsp_stub(app, client):
    _login(client)
    resp = client.post(
        "/cameras/demo-rtsp-01/enabled",
        data={"enabled": "0"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert app.extensions["camera_store"].get("demo-rtsp-01").enabled is False


def test_run_registered_file_camera_creates_stamped_alerts(app, client):
    _login(client)
    resp = client.post(
        "/run",
        data={"mode": "camera", "camera_id": "demo-file-01"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "alert(s) queued for human review" in body
    assert "Category summary" in body
    assert "game or play" in body
    assert "dance" in body
    assert "potential confrontation" in body
    store = app.extensions["alert_store"]
    alerts = store.list_alerts()
    assert alerts
    assert all(a.camera_id == "demo-file-01" for a in alerts)
    assert all(a.location_label == "Demo Lab" for a in alerts)
    assert "Place type" in body
    assert "gymnasium" in body.lower()
    stamped = [
        a.to_dict()["metadata"]
        for a in alerts
        if a.to_dict().get("metadata", {}).get("place_type")
    ]
    assert stamped
    assert all(meta.get("place_type") == "gymnasium" for meta in stamped)
    assert all("madison" not in (a.rationale or "").lower() for a in alerts)
    cam = app.extensions["camera_store"].get("demo-file-01")
    assert cam.last_seen_at
    assert cam.last_error is None


def test_run_disabled_rtsp_stub_records_error_not_alerts(app, client, monkeypatch):
    monkeypatch.delenv("RTSP_DEMO_URI", raising=False)
    cameras = app.extensions["camera_store"]
    cameras.set_enabled("demo-rtsp-01", True)
    before = len(app.extensions["alert_store"].list_alerts())
    _login(client)
    resp = client.post(
        "/run",
        data={"mode": "camera", "camera_id": "demo-rtsp-01"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Pipeline error" not in body or "honest" in body.lower() or "error" in body.lower()
    after = app.extensions["alert_store"].list_alerts()
    new = [a for a in after if a.camera_id == "demo-rtsp-01"]
    assert new == []
    assert len(after) == before
    err = cameras.get("demo-rtsp-01").last_error
    assert err
    assert "No detections generated" in err or "missing" in err.lower() or "empty" in err.lower()


def test_dashboard_shows_camera_health(client):
    _login(client)
    resp = client.get("/")
    body = resp.get_data(as_text=True)
    assert "Camera health" in body
    assert "Demo Lab File" in body
    assert "Cameras" in body


def test_run_synthetic_shows_game_dance_without_threat_alerts(app, client):
    _login(client)
    resp = client.post("/run", data={"mode": "synthetic"}, follow_redirects=True)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Category summary" in body
    assert "Scene assists" in body
    assert "game or play" in body
    assert "dance" in body
    assert "potential confrontation" in body
    assert "Basketball" in body
    assert "Sport context" in body
    assert "Place type" in body
    assert "Kit cues" in body
    assert "Body-aggression" in body
    assert "Face cue status" in body
    assert "Fall manner" in body
    assert "Gunshot video proxy" in body
    assert "object_thrown_at_person" in body
    assert "firearm_aimed_at_person" in body
    assert "disabled" in body
    categories = {a.category for a in app.extensions["alert_store"].list_alerts()}
    assert "game_or_play" not in categories
    assert "dance" not in categories
    assert "potential_fight" in categories


def test_camera_form_lists_console_users_and_assigns_owner(app, client):
    _login(client)
    admin = app.extensions["user_store"].get_by_username("operator")
    assert admin is not None
    form = client.get("/cameras/new")
    body = form.get_data(as_text=True)
    assert form.status_code == 200
    assert "Accounts linked to this camera" in body
    assert admin.email in body
    assert "operator" in body
    assert "My cameras" in body
    created = client.post(
        "/cameras/new",
        data={
            "name": "Owned hall",
            "location_label": "North corridor",
            "source_type": "file",
            "uri": "MOCK",
            "sample_fps": "2",
            "enabled": "1",
            "place_type": "corridor_hallway",
            "account_user_ids": [admin.id],
            "notify_email": "security@example.com",
            "notes": "Authorized hall",
        },
        follow_redirects=True,
    )
    assert created.status_code == 200
    listing = created.get_data(as_text=True)
    assert "Owned hall" in listing
    assert "operator" in listing
    stored = [
        c for c in app.extensions["camera_store"].list_cameras() if c.name == "Owned hall"
    ]
    assert stored
    assert stored[0].owner_user_id == admin.id
    assert stored[0].owner_username == "operator"
    assert stored[0].notify_email == "security@example.com"
    links = app.extensions["camera_store"].list_accounts_for_camera(stored[0].id)
    assert {row["user_id"] for row in links} == {admin.id}
    edit = client.get(f"/cameras/{stored[0].id}/edit")
    edit_body = edit.get_data(as_text=True)
    assert f'value="{admin.id}"' in edit_body
    assert "checked" in edit_body
    cleared = client.post(
        f"/cameras/{stored[0].id}/edit",
        data={
            "name": "Owned hall",
            "location_label": "North corridor",
            "source_type": "file",
            "uri": "",
            "sample_fps": "2",
            "enabled": "1",
            "place_type": "corridor_hallway",
            "notify_email": "",
            "notes": "Authorized hall",
        },
        follow_redirects=True,
    )
    assert cleared.status_code == 200
    refreshed = app.extensions["camera_store"].get(stored[0].id)
    assert refreshed.owner_user_id == ""
    assert refreshed.notify_email == ""
    assert app.extensions["camera_store"].list_accounts_for_camera(refreshed.id) == []


def test_camera_form_has_place_type_dropdown(client):
    _login(client)
    resp = client.get("/cameras/new")
    body = resp.get_data(as_text=True)
    assert "Place type" in body
    assert "corridor_hallway" in body
    assert "house_interior" in body
    assert "compound_courtyard" in body
    assert "roam" in body
    assert "Roam / patrol" in body
    assert "Madison Square Garden" not in body

    created = client.post(
        "/cameras/new",
        data={
            "name": "Hall cam",
            "location_label": "North corridor",
            "source_type": "file",
            "uri": "MOCK",
            "sample_fps": "2",
            "enabled": "1",
            "place_type": "corridor_hallway",
            "notes": "Authorized hall",
        },
        follow_redirects=True,
    )
    assert created.status_code == 200
    assert "Hall cam" in created.get_data(as_text=True)
    assert "corridor_hallway" in created.get_data(as_text=True)
    assert "Other / custom" in body


def test_camera_form_custom_place_round_trip(app, client):
    _login(client)
    created = client.post(
        "/cameras/new",
        data={
            "name": "Roof café cam",
            "location_label": "Roof terrace",
            "source_type": "file",
            "uri": "MOCK",
            "sample_fps": "2",
            "enabled": "1",
            "place_type": "__custom__",
            "place_type_custom": "rooftop café",
            "notes": "Authorized rooftop",
        },
        follow_redirects=True,
    )
    assert created.status_code == 200
    listing = created.get_data(as_text=True)
    assert "Roof café cam" in listing
    assert "rooftop_cafe" in listing
    stored = [
        c
        for c in app.extensions["camera_store"].list_cameras()
        if c.name == "Roof café cam"
    ]
    assert stored
    assert stored[0].place_type == "rooftop_cafe"
    edit = client.get(f"/cameras/{stored[0].id}/edit")
    edit_body = edit.get_data(as_text=True)
    assert edit.status_code == 200
    assert "rooftop_cafe" in edit_body
    assert "Other / custom" in edit_body


def test_run_page_lists_registry(client):
    _login(client)
    resp = client.get("/run")
    body = resp.get_data(as_text=True)
    assert "Registered camera" in body
    assert "demo-file-01" in body
    assert "All enabled cameras" in body
    assert "Authorized video file upload" in body
    assert "video-file" in body
    assert "mode-upload" in body
    assert "upload-hint" in body


def _tiny_video(path: Path, frames: int = 40) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter MJPG unavailable in this environment")
    for i in range(frames):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[:] = ((i * 20) % 255, 40, 90)
        writer.write(img)
    writer.release()
    return path


class _OrdinaryAdapter(VisionAdapter):
    name = "ordinary-test"

    def detect(self, image_bgr, frame_index: int = 0):
        return [Detection("ordinary", 0.95)]


def test_run_synthetic_mode_with_file_uses_upload(app, client, tmp_path):
    video = _tiny_video(tmp_path / "src" / "hallway.avi")
    _login(client)
    resp = client.post(
        "/run",
        data={
            "mode": "synthetic",
            "video": (BytesIO(video.read_bytes()), "hallway.avi"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "authorized upload — hallway.avi" in body
    assert "hallway.avi" in body
    assert "Synthetic Demo" not in body
    assert "Authorized video file upload" in body
    assert "form mode was synthetic" in body
    frames = re.search(r"Frames processed:\s*(\d+)", body)
    assert frames and int(frames.group(1)) > 0
    saved = Path(app.config["UPLOAD_DIR"]) / "hallway.avi"
    assert saved.is_file()
    alerts = app.extensions["alert_store"].list_alerts()
    assert all("authorized upload — hallway.avi" in a.source_label for a in alerts)
    assert all("Synthetic" not in a.source_label for a in alerts)


def test_run_camera_mode_with_file_uses_upload(app, client, tmp_path):
    video = _tiny_video(tmp_path / "src" / "gate.avi", frames=12)
    _login(client)
    resp = client.post(
        "/run",
        data={
            "mode": "camera",
            "camera_id": "demo-file-01",
            "video": (BytesIO(video.read_bytes()), "gate.avi"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "authorized upload — gate.avi" in body
    assert "form mode was camera" in body
    frames = re.search(r"Frames processed:\s*(\d+)", body)
    assert frames and int(frames.group(1)) > 0
    alerts = app.extensions["alert_store"].list_alerts()
    assert all("authorized upload" in a.source_label for a in alerts)


def test_upload_quiet_run_explains_zero_alerts(app, client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vision.detector.create_adapter", lambda *a, **k: _OrdinaryAdapter()
    )
    video = _tiny_video(tmp_path / "src" / "ordinary.avi", frames=12)
    _login(client)
    before = len(app.extensions["alert_store"].list_alerts())
    resp = client.post(
        "/run",
        data={
            "mode": "upload",
            "video": (BytesIO(video.read_bytes()), "ordinary.avi"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "authorized upload — ordinary.avi" in body
    assert "no elevated-risk frames" in body
    assert "ordinary" in body
    assert "Top predicted categories" in body
    frames = re.search(r"Frames processed:\s*(\d+)", body)
    assert frames and int(frames.group(1)) > 0
    assert len(app.extensions["alert_store"].list_alerts()) == before


def test_run_synthetic_without_file_still_uses_mock(app, client):
    _login(client)
    resp = client.post("/run", data={"mode": "synthetic"}, follow_redirects=True)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Synthetic Demo" in body
    alerts = app.extensions["alert_store"].list_alerts()
    assert alerts
    assert all("Synthetic" in a.source_label for a in alerts)


def test_factory_creates_upload_dir(app):
    upload_dir = Path(app.config["UPLOAD_DIR"])
    assert upload_dir.is_dir()
