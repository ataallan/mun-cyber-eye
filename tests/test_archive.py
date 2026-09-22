"""Optional video archive: off by default, retention, and honest failures."""

import json
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from alerts.store import AlertStore
from app.factory import create_app
from ingest.archive import (
    ArchiveDiskFull,
    ArchiveError,
    ArchiveSkipped,
    ArchiveStore,
    SegmentWriter,
    adopt_finished_files,
    clamp_retention_days,
    clamp_segment_sec,
    disk_cost_note,
    format_utc,
    open_archive_capture,
    prune_archive,
    utc_now,
)
from ingest.sampler import SampledFrame


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
            "ARCHIVE_DIR": str(tmp_path / "archive"),
            "ARCHIVE_DB_PATH": str(tmp_path / "archive.db"),
            "ARCHIVE_STATE_PATH": str(tmp_path / "archive_state.json"),
            "MONITOR_STATE_PATH": str(tmp_path / "monitor_state.json"),
            "VISION_BACKEND": "mock",
            "ALERT_NOTIFY_ON_CREATE": False,
            "ALLOW_WEBCAM": False,
            "SEED_DEMO_CAMERAS": False,
            "MONITOR_AUTOSTART": False,
            "MONITOR_CYCLE_PAUSE_SEC": 0.2,
            "ARCHIVE_MIN_FREE_BYTES": 1,
            "ARCHIVE_ENABLED": False,
        }
    )
    yield application
    application.extensions["monitor"].stop(join_timeout=5)
    application.extensions["archive"].shutdown(timeout=5)


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client, username="siteadmin", password="test-pass-12"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


def _write_tiny_video(path: Path, n_frames: int = 8, fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (64, 48),
    )
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = ((i * 24) % 255, 40, 90)
        writer.write(frame)
    writer.release()


def _videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".avi", ".webm"}
        and not path.name.endswith(".part.avi")
    ]


def _wait(predicate, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _status_text(status: dict) -> str:
    parts = [status.get("last_error") or ""]
    parts.extend(item.get("message") or "" for item in status.get("errors") or [])
    return " ".join(parts)


class _DeadCapture:
    def isOpened(self):
        return True

    def get(self, _prop):
        return 8.0

    def read(self):
        return False, None

    def release(self):
        return None


def test_retention_bounds_and_disk_note():
    assert clamp_retention_days(1) == 1
    assert clamp_retention_days(5) == 5
    assert clamp_retention_days(0) == 1
    assert clamp_retention_days(90) == 90
    assert clamp_retention_days(91) == 1
    assert clamp_segment_sec(30) == 60
    assert clamp_segment_sec(120) == 120
    assert clamp_segment_sec(400) == 300
    note = disk_cost_note(8)
    assert "GB" in note
    assert "hour" in note


def test_mock_webcam_and_missing_file_do_not_invent_footage(tmp_path):
    class Camera:
        def __init__(self, source_type, uri):
            self.source_type = source_type
            self.uri = uri

    with pytest.raises(ArchiveSkipped):
        open_archive_capture(Camera("file", "MOCK"), project_root=tmp_path)
    with pytest.raises(ArchiveError, match="No archive footage was recorded"):
        open_archive_capture(
            Camera("file", "missing-feed.avi"),
            project_root=tmp_path,
            allow_webcam=False,
        )
    with pytest.raises(ArchiveError, match="No archive footage was recorded"):
        open_archive_capture(Camera("webcam", "0"), allow_webcam=False)
    assert _videos(tmp_path) == []


def test_segment_rolls_and_disk_full_writes_nothing(tmp_path):
    store = ArchiveStore(tmp_path / "archive.db")
    root = tmp_path / "archive"
    moment = utc_now()
    clock = {"now": moment}

    def _clock():
        return clock["now"]

    writer = SegmentWriter(
        root,
        "gate-1",
        store,
        fps=8,
        segment_sec=1,
        min_free_bytes=0,
        clock=_clock,
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)
    writer.write(frame)
    clock["now"] = moment + timedelta(seconds=2)
    frame[:] = (80, 90, 100)
    writer.write(frame)
    writer.close()
    finished = _videos(root)
    assert len(finished) == 2
    assert len(store.list_segments()) == 2
    assert all(path.parent.name == "gate-1" for path in finished)

    blocked = SegmentWriter(
        tmp_path / "full",
        "gate-1",
        store,
        min_free_bytes=10**18,
    )
    with pytest.raises(ArchiveDiskFull, match="No footage was invented"):
        blocked.write(frame)
    assert _videos(tmp_path / "full") == []


def test_prune_keeps_window_and_leaves_alerts_and_clips(tmp_path):
    now = utc_now()
    root = tmp_path / "archive"
    store = ArchiveStore(tmp_path / "archive.db")
    clips = tmp_path / "clips"
    clips.mkdir()
    clip = clips / "incident.mp4"
    clip.write_bytes(b"incident-clip")

    def _add(camera_id: str, name: str, age: timedelta) -> None:
        folder = root / camera_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_bytes(b"segment-bytes")
        ended = now - age
        started = ended - timedelta(seconds=30)
        store.add_segment(
            camera_id=camera_id,
            relpath=f"{camera_id}/{name}",
            started_at=format_utc(started),
            ended_at=format_utc(ended),
            nbytes=path.stat().st_size,
        )

    _add("cam-a", "recent.mp4", timedelta(hours=1))
    _add("cam-a", "two-days.mp4", timedelta(days=2))
    _add("cam-a", "six-days.mp4", timedelta(days=6))
    old_orphan = root / "cam-a" / "orphan-old.mp4"
    old_orphan.write_bytes(b"old-orphan")
    old_ts = (now - timedelta(days=10)).timestamp()
    old_orphan.touch()
    import os

    os.utime(old_orphan, (old_ts, old_ts))
    recent_orphan = root / "cam-a" / "orphan-new.mp4"
    recent_orphan.write_bytes(b"new-orphan")

    alerts = AlertStore(tmp_path / "alerts.db")
    alert = alerts.create_alert(
        source_label="Authorized Camera — Archive",
        category="potential_fight",
        risk_level="high",
        confidence=0.8,
        rationale="Keep this alert row.",
        frame_index=1,
        timestamp_sec=1.0,
        clip_path=str(clip),
    )

    removed = prune_archive(store, root, retention_days=5, now=now)
    assert removed >= 2
    names = {path.name for path in _videos(root)}
    assert "six-days.mp4" not in names
    assert "orphan-old.mp4" not in names
    assert "two-days.mp4" in names
    assert "recent.mp4" in names
    assert recent_orphan.is_file()
    assert clip.is_file()
    assert alerts.get(alert.id) is not None

    prune_archive(store, root, retention_days=1, now=now)
    names = {path.name for path in _videos(root)}
    assert "two-days.mp4" not in names
    assert "recent.mp4" in names
    assert clip.is_file()
    assert alerts.get(alert.id) is not None
    assert alerts.get(alert.id).clip_path == str(clip)


def test_startup_indexes_finished_segment(tmp_path):
    root = tmp_path / "archive" / "gate-1"
    root.mkdir(parents=True)
    path = root / "20260922T040000Z_ab12cd.mp4"
    path.write_bytes(b"finished-segment")
    store = ArchiveStore(tmp_path / "archive.db")
    added = adopt_finished_files(store, tmp_path / "archive")
    assert added == 1
    rows = store.list_segments(camera_id="gate-1")
    assert len(rows) == 1
    assert rows[0].started_at.startswith("2026-09-22T04:00:00")


def test_archive_off_writes_no_continuous_files(app):
    video = Path(app.config["SNAPSHOT_DIR"]) / "source.avi"
    _write_tiny_video(video)
    cameras = app.extensions["camera_store"]
    cameras.create(
        camera_id="gate-1",
        name="Gate",
        location_label="North",
        source_type="file",
        uri=str(video),
        enabled=True,
    )
    called = {"n": 0}

    def _opener(_camera):
        called["n"] += 1
        raise AssertionError("archive opened a camera while off")

    archive = app.extensions["archive"]
    archive.capture_opener = _opener
    assert archive.status()["enabled"] is False
    app.extensions["monitor"].start()
    time.sleep(0.6)
    app.extensions["monitor"].stop(join_timeout=5)
    assert called["n"] == 0
    assert _videos(Path(app.config["ARCHIVE_DIR"])) == []
    assert archive.status()["recording"] is False


def test_archive_records_enabled_camera_apart_from_clips(app):
    video = Path(app.config["SNAPSHOT_DIR"]) / "source.avi"
    _write_tiny_video(video)
    cameras = app.extensions["camera_store"]
    cameras.create(
        camera_id="gate-1",
        name="Gate",
        location_label="North",
        source_type="file",
        uri=str(video),
        enabled=True,
    )
    cameras.create(
        camera_id="gate-off",
        name="Side",
        location_label="South",
        source_type="file",
        uri=str(video),
        enabled=False,
    )
    archive = app.extensions["archive"]
    archive.set_enabled(True)
    assert _videos(Path(app.config["ARCHIVE_DIR"])) == []
    app.extensions["monitor"].start()
    assert _wait(lambda: bool(_videos(Path(app.config["ARCHIVE_DIR"]))))
    app.extensions["monitor"].stop(join_timeout=5)
    archive.set_enabled(False)
    videos = _videos(Path(app.config["ARCHIVE_DIR"]))
    assert videos
    archive_root = Path(app.config["ARCHIVE_DIR"]).resolve()
    clip_root = Path(app.config["CLIP_DIR"]).resolve()
    assert all(archive_root in path.resolve().parents for path in videos)
    assert all(clip_root not in path.resolve().parents for path in videos)
    assert any(path.parent.name == "gate-1" for path in videos)
    assert not (archive_root / "gate-off").exists()
    rows = archive.store.list_segments(camera_id="gate-1")
    assert rows
    assert all(row.camera_id == "gate-1" for row in rows)
    for clip in _videos(clip_root):
        assert archive_root not in clip.resolve().parents


def test_dead_stream_and_full_disk_are_reported(app, tmp_path):
    cameras = app.extensions["camera_store"]
    cameras.create(
        camera_id="dead-rtsp",
        name="Dead feed",
        source_type="rtsp",
        uri="rtsp://127.0.0.1:9/dead",
        enabled=True,
    )
    archive = app.extensions["archive"]
    archive.capture_opener = lambda _camera: _DeadCapture()
    archive.set_enabled(True)
    app.extensions["monitor"].start()
    assert _wait(
        lambda: "No archive footage was recorded" in _status_text(archive.status())
    )
    assert _videos(Path(app.config["ARCHIVE_DIR"])) == []
    app.extensions["monitor"].stop(join_timeout=5)
    archive.set_enabled(False)

    video = tmp_path / "source.avi"
    _write_tiny_video(video)
    cameras.create(
        camera_id="full-disk",
        name="Full",
        source_type="file",
        uri=str(video),
        enabled=True,
    )
    cameras.set_enabled("dead-rtsp", False)
    app.config["ARCHIVE_MIN_FREE_BYTES"] = 10**18
    archive.capture_opener = None
    archive.set_enabled(True)
    app.extensions["monitor"].start()
    assert _wait(lambda: "Disk is full" in _status_text(archive.status()))
    assert "No footage was invented" in _status_text(archive.status())
    assert _videos(Path(app.config["ARCHIVE_DIR"])) == []
    app.extensions["monitor"].stop(join_timeout=5)


def test_console_retention_and_playback(app, client):
    anon = client.get("/archive")
    assert anon.status_code == 302
    assert "/login" in anon.headers["Location"]

    _login(client)
    page = client.get("/archive")
    body = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "1 day" in body
    assert "5 days" in body
    assert "No archive segments" in body

    saved = client.post(
        "/archive/settings",
        data={"retention_days": "5", "next": "archive"},
        follow_redirects=True,
    )
    assert app.extensions["archive"].status()["retention_days"] == 5
    assert "5 day" in saved.get_data(as_text=True)
    custom = client.post(
        "/archive/settings",
        data={"retention_days": "5", "retention_days_custom": "1", "next": "archive"},
        follow_redirects=True,
    )
    assert custom.status_code == 200
    assert app.extensions["archive"].status()["retention_days"] == 1
    rejected = client.post(
        "/archive/settings",
        data={"retention_days_custom": "0", "next": "archive"},
        follow_redirects=True,
    )
    assert "1 to 90" in rejected.get_data(as_text=True)
    assert app.extensions["archive"].status()["retention_days"] == 1

    started = client.post("/archive/start", data={"next": "archive"}, follow_redirects=True)
    assert "Stop archive" in started.get_data(as_text=True)
    assert app.extensions["archive"].status()["enabled"] is True
    state = json.loads(Path(app.config["ARCHIVE_STATE_PATH"]).read_text(encoding="utf-8"))
    assert state["enabled"] is True
    assert state["retention_days"] == 1

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[:] = (12, 40, 80)
    writer = SegmentWriter(
        app.config["ARCHIVE_DIR"],
        "gate-1",
        app.extensions["archive"].store,
        fps=8,
        segment_sec=30,
        min_free_bytes=0,
    )
    writer.write(frame)
    writer.close()
    review = client.get("/archive?window=1&camera_id=gate-1")
    review_body = review.get_data(as_text=True)
    assert "Play" in review_body
    assert "gate-1" in review_body
    segment = app.extensions["archive"].store.list_segments(camera_id="gate-1")[0]
    media = client.get(f"/archive/media/{segment.id}")
    assert media.status_code == 200
    assert media.data

    outsider = app.test_client()
    hidden = outsider.get(f"/archive/media/{segment.id}")
    assert hidden.status_code == 302
    assert "/login" in hidden.headers["Location"]
    missing = client.get("/archive/media/not-a-real-id")
    assert missing.status_code == 404

    stopped = client.post("/archive/stop", data={"next": "archive"}, follow_redirects=True)
    assert "Start archive" in stopped.get_data(as_text=True)
    assert app.extensions["archive"].status()["enabled"] is False


def test_sampled_clip_helper_is_not_the_archive_path(tmp_path):
    """Incident-clip frames are a different API from archive segments."""
    from ingest.clips import write_incident_clip

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    frames = [
        SampledFrame(index=0, timestamp_sec=0.0, image_bgr=img, source_label="clip"),
        SampledFrame(index=1, timestamp_sec=0.5, image_bgr=img, source_label="clip"),
    ]
    clip = write_incident_clip(frames, tmp_path / "clips", camera_id="gate-1")
    assert clip is not None
    assert clip.parent == tmp_path / "clips"
    assert _videos(tmp_path / "archive") == []
