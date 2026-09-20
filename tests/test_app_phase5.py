"""Flask console: camera registry, run-from-camera, webcam gate, health."""

import pytest

from app.factory import create_app


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
    store = app.extensions["alert_store"]
    alerts = store.list_alerts()
    assert alerts
    assert all(a.camera_id == "demo-file-01" for a in alerts)
    assert all(a.location_label == "Demo Lab" for a in alerts)
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


def test_run_page_lists_registry(client):
    _login(client)
    resp = client.get("/run")
    body = resp.get_data(as_text=True)
    assert "Registered camera" in body
    assert "demo-file-01" in body
    assert "All enabled cameras" in body
