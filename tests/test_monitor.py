"""Continuous monitoring, incident clips, and honest failures."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from alerts.store import AlertStore
from app.factory import create_app
from ingest.clips import prune_incident_clips, write_incident_clip
from ingest.errors import IngestError
from ingest.sampler import SampledFrame
from pipeline import CyberEyePipeline
from vision.detector import Detection, VisionAdapter


@pytest.fixture
def app(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_ROLE": "admin",
            "ALERT_DB_PATH": str(tmp_path / "app.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snaps"),
            "CLIP_DIR": str(tmp_path / "clips"),
            "MONITOR_STATE_PATH": str(tmp_path / "monitor_state.json"),
            "VISION_BACKEND": "mock",
            "ALERT_NOTIFY_ON_CREATE": True,
            "ALLOW_WEBCAM": False,
            "RTSP_CONNECT_TIMEOUT_SEC": 1,
            "MONITOR_AUTOSTART": False,
            "MONITOR_CYCLE_PAUSE_SEC": 0.2,
            "MONITOR_ALERT_COOLDOWN_SEC": 60,
        }
    )
    yield application
    application.extensions["monitor"].stop(join_timeout=5)


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client, username="siteadmin", password="test-pass-12"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


def _disable_all(app):
    cameras = app.extensions["camera_store"]
    for camera in cameras.list_cameras():
        cameras.set_enabled(camera.id, False)


class _ThreatAt(VisionAdapter):
    name = "threat-at"

    def __init__(self, index: int) -> None:
        self.index = index

    def detect(self, image_bgr, frame_index: int = 0):
        if frame_index == self.index:
            return [
                Detection("person", 0.90, (80, 40, 220, 420)),
                Detection("person", 0.89, (150, 50, 290, 430)),
                Detection("close_proximity", 0.78, (80, 40, 290, 430)),
                Detection("rapid_motion", 0.82, (100, 80, 280, 380)),
            ]
        return [Detection("person", 0.90, (40, 50, 180, 400))]


def _frames(n: int, step: float = 0.5):
    frames = []
    for i in range(n):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[:, :, 1] = (i * 8) % 255
        img[:, :, 2] = 40
        frames.append(
            SampledFrame(
                index=i,
                timestamp_sec=i * step,
                image_bgr=img,
                source_label="Authorized Camera — Clip Lab",
            )
        )
    return frames


def test_testing_app_does_not_autostart(app):
    status = app.extensions["monitor"].status()
    assert status["running"] is False
    assert status["desired"] == "off"


def test_monitor_start_stop_from_console(app, client):
    _disable_all(app)
    anon = client.post("/monitor/start")
    assert anon.status_code == 302
    assert "/login" in anon.headers["Location"]

    _login(client)
    started = client.post("/monitor/start", follow_redirects=True)
    body = started.get_data(as_text=True)
    assert started.status_code == 200
    assert "Stop monitoring" in body
    assert app.extensions["monitor"].status()["running"] is True
    assert "Monitoring" in body

    stopped = client.post("/monitor/stop", follow_redirects=True)
    stopped_body = stopped.get_data(as_text=True)
    assert "Start monitoring" in stopped_body
    assert app.extensions["monitor"].status()["running"] is False


def test_monitor_saves_playable_clip_and_notifies(app, client):
    _disable_all(app)
    cameras = app.extensions["camera_store"]
    cameras.create(
        camera_id="clip-lab",
        name="Clip Lab",
        location_label="West stair",
        source_type="file",
        uri="MOCK",
        notify_email="security@example.com",
        enabled=True,
    )
    summary = app.extensions["monitor"].run_once()
    assert summary["failures"] == 0
    assert summary["alerts"] >= 1
    alerts = [
        alert
        for alert in app.extensions["alert_store"].list_alerts()
        if alert.camera_id == "clip-lab"
    ]
    assert alerts
    clipped = [alert for alert in alerts if alert.clip_path]
    assert clipped
    clip = Path(clipped[0].clip_path)
    assert clip.is_file()
    assert clip.parent == Path(app.config["CLIP_DIR"])
    capture = cv2.VideoCapture(str(clip))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok and frame is not None

    refreshed = app.extensions["alert_store"].get(clipped[0].id)
    assert refreshed.delivery_status in {"queued", "undelivered", "sent", "partial"}
    audit = app.extensions["alert_store"].audit_trail(clipped[0].id)
    assert any(row["action"] == "notify" for row in audit)
    payload = refreshed.to_dict()
    assert payload["clip_path"]
    assert payload["metadata"]["incident_clip"]["duration_sec"] >= 0

    again = app.extensions["monitor"].run_once()
    assert again["alerts"] == 0
    assert (
        len(
            [
                alert
                for alert in app.extensions["alert_store"].list_alerts()
                if alert.camera_id == "clip-lab"
            ]
        )
        == len(alerts)
    )

    _login(client)
    page = client.get(f"/alerts/{clipped[0].id}")
    html = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "<video" in html
    clip_resp = client.get(f"/clips/{clip.name}")
    assert clip_resp.status_code == 200
    assert len(clip_resp.data) > 32
    anon = app.test_client().get(f"/clips/{clip.name}")
    assert anon.status_code == 302


def test_failed_streams_do_not_invent_detections(app):
    _disable_all(app)
    cameras = app.extensions["camera_store"]
    missing = cameras.create(
        camera_id="missing-file",
        name="Missing file",
        location_label="Dock",
        source_type="file",
        uri=str(Path(app.config["CLIP_DIR"]) / "no-such-feed.mp4"),
        enabled=True,
    )
    dead = cameras.create(
        camera_id="dead-rtsp",
        name="Dead RTSP",
        location_label="Fence",
        source_type="rtsp",
        uri="rtsp://user:secret@127.0.0.1:1/offline",
        enabled=True,
    )

    def fail_open(uri, timeout_sec):
        raise IngestError(
            "Unable to open authorized RTSP feed (camera offline). "
            "No detections generated."
        )

    monitor = app.extensions["monitor"]
    monitor.rtsp_opener = fail_open
    before = len(app.extensions["alert_store"].list_alerts())
    summary = monitor.run_once()
    assert summary["alerts"] == 0
    assert summary["failures"] == 2
    assert len(app.extensions["alert_store"].list_alerts()) == before
    missing_row = cameras.get(missing.id)
    dead_row = cameras.get(dead.id)
    assert missing_row.last_error
    assert "No detections generated" in missing_row.last_error
    assert dead_row.last_error
    assert "No detections generated" in dead_row.last_error
    assert "secret" not in (dead_row.last_error or "")


def test_clip_window_and_cooldown(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    calls = []

    class _Spy:
        def deliver(self, alert, actor="system", extra_recipients=None):
            calls.append(alert.id)
            return alert

    pipe = CyberEyePipeline(
        store=store,
        adapter=_ThreatAt(10),
        snapshot_dir=tmp_path / "snaps",
        notifier=_Spy(),
        clip_dir=tmp_path / "clips",
        clip_pre_sec=5,
        clip_post_sec=5,
        clip_max_sec=15,
        alert_cooldown_sec=60,
        object_mode="off",
    )
    first = pipe.run_frames(
        _frames(25, step=0.5),
        source_label="Authorized Camera — Clip Lab",
        camera_id="cam-clip",
        location_label="West stair",
    )
    assert len(first.alerts_created) == 1
    alert = first.alerts_created[0]
    assert alert.camera_id == "cam-clip"
    assert alert.clip_path
    meta = alert.to_dict()["metadata"]["incident_clip"]
    assert 8.0 <= meta["duration_sec"] <= 12.0
    assert calls == [alert.id]

    second = pipe.run_frames(
        _frames(25, step=0.5),
        source_label="Authorized Camera — Clip Lab",
        camera_id="cam-clip",
        location_label="West stair",
    )
    assert second.alerts_created == []
    assert len(store.list_alerts()) == 1


def test_one_shot_run_does_not_require_a_clip(app, client):
    _login(client)
    client.post(
        "/run",
        data={"mode": "camera", "camera_id": "demo-file-01"},
        follow_redirects=True,
    )
    alerts = [
        alert
        for alert in app.extensions["alert_store"].list_alerts()
        if alert.camera_id == "demo-file-01"
    ]
    assert alerts
    assert all(not alert.clip_path for alert in alerts)


def test_prune_keeps_newest_clips_only(tmp_path):
    folder = tmp_path / "clips"
    written = []
    for index in range(3):
        frame = _frames(2)[index % 2]
        path = write_incident_clip(
            [frame, frame],
            folder,
            camera_id="cam",
            category="potential_fight",
        )
        assert path is not None
        written.append(path)
    removed = prune_incident_clips(folder, max_clips=1)
    assert removed == 2
    remaining = [path for path in written if path.exists()]
    assert len(remaining) == 1


def test_saved_stop_beats_autostart(tmp_path):
    common = {
        "TESTING": True,
        "SECRET_KEY": "test",
        "ADMIN_USERNAME": "siteadmin",
        "ADMIN_PASSWORD": "test-pass-12",
        "ALERT_DB_PATH": str(tmp_path / "app.db"),
        "AUTH_DB_PATH": str(tmp_path / "auth.db"),
        "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
        "SNAPSHOT_DIR": str(tmp_path / "snaps"),
        "CLIP_DIR": str(tmp_path / "clips"),
        "MONITOR_STATE_PATH": str(tmp_path / "monitor_state.json"),
        "VISION_BACKEND": "mock",
        "ALERT_NOTIFY_ON_CREATE": False,
        "SEED_DEMO_CAMERAS": False,
        "MONITOR_AUTOSTART": True,
        "MONITOR_CYCLE_PAUSE_SEC": 0.2,
    }
    first = create_app(common)
    try:
        assert first.extensions["monitor"].status()["running"] is True
        first.extensions["monitor"].stop(join_timeout=5)
        assert first.extensions["monitor"].status()["running"] is False
    finally:
        first.extensions["monitor"].stop(join_timeout=5)

    second = create_app(common)
    try:
        assert second.extensions["monitor"].status()["running"] is False
        assert second.extensions["monitor"].status()["desired"] == "off"
    finally:
        second.extensions["monitor"].stop(join_timeout=5)
