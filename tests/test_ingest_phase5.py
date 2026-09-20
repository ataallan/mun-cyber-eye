"""File / RTSP / webcam ingest and camera-stamped alerts."""

from pathlib import Path

import numpy as np
import pytest

from alerts.store import AlertStore
from ingest.cameras import CameraStore
from ingest.errors import IngestError
from ingest.rtsp import RtspSampler
from ingest.sampler import WebcamStub
from ingest.source import iter_camera_frames
from ingest.webcam import WEBCAM_REFUSED_MESSAGE, WebcamSampler
from pipeline import CyberEyePipeline, run_registered_cameras
from vision.detector import MockVisionAdapter


@pytest.fixture
def stores(tmp_path):
    cameras = CameraStore(tmp_path / "cameras.db")
    alerts = AlertStore(tmp_path / "alerts.db")
    return cameras, alerts, tmp_path


def _tiny_video(path: Path, frames: int = 10) -> Path:
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


def test_file_camera_run_stamps_alert_metadata(stores):
    cameras, alerts, tmp = stores
    cam = cameras.create(
        camera_id="lab-file",
        name="Demo Lab File",
        location_label="Demo Lab East",
        source_type="file",
        uri="MOCK",
        sample_fps=2.0,
    )
    pipe = CyberEyePipeline(
        store=alerts,
        adapter=MockVisionAdapter(),
        snapshot_dir=tmp / "snaps",
    )
    result = pipe.run_camera(cam, max_frames=16, project_root=tmp)
    cameras.record_success(cam.id)
    assert result.frames_processed == 16
    assert result.camera_id == "lab-file"
    assert result.location_label == "Demo Lab East"
    assert len(result.alerts_created) >= 1
    for alert in result.alerts_created:
        assert alert.camera_id == "lab-file"
        assert alert.location_label == "Demo Lab East"
        assert "Demo Lab File" in alert.source_label
    refreshed = cameras.get(cam.id)
    assert refreshed.last_error is None
    assert refreshed.last_seen_at


def test_real_file_camera_samples_frames(stores):
    cameras, alerts, tmp = stores
    video = _tiny_video(tmp / "clip.avi")
    cam = cameras.create(
        name="File clip",
        location_label="Loading dock",
        source_type="file",
        uri=str(video),
        sample_fps=5.0,
    )
    pipe = CyberEyePipeline(
        store=alerts,
        adapter=MockVisionAdapter(),
        snapshot_dir=tmp / "snaps",
    )
    result = pipe.run_camera(cam, max_frames=8, project_root=tmp)
    assert result.frames_processed >= 1
    assert result.location_label == "Loading dock"
    if result.alerts_created:
        assert result.alerts_created[0].camera_id == cam.id


def test_rtsp_failure_marks_last_error_without_alerts(stores):
    cameras, alerts, tmp = stores
    cam = cameras.create(
        name="Perimeter stub",
        location_label="Fence",
        source_type="rtsp",
        uri="rtsp://user:pass@127.0.0.1:1/offline",
        enabled=True,
    )

    def fail_open(uri, timeout_sec):
        raise IngestError(
            "Unable to open authorized RTSP feed (timeout 1s or camera offline). "
            "No detections generated."
        )

    pipe = CyberEyePipeline(
        store=alerts,
        adapter=MockVisionAdapter(),
        snapshot_dir=tmp / "snaps",
    )
    results = run_registered_cameras(
        pipe,
        [cam],
        cameras,
        max_frames=4,
        project_root=tmp,
        timeout_sec=1,
        rtsp_opener=fail_open,
    )
    assert len(results) == 1
    assert results[0].error
    assert results[0].alerts_created == []
    assert alerts.list_alerts() == []
    refreshed = cameras.get(cam.id)
    assert refreshed.last_error
    assert "No detections generated" in refreshed.last_error
    assert refreshed.last_seen_at is None


def test_rtsp_sampler_missing_uri():
    with pytest.raises(IngestError, match="No detections generated"):
        list(RtspSampler("").frames())


def test_webcam_refused_when_allow_webcam_off(monkeypatch):
    monkeypatch.setenv("ALLOW_WEBCAM", "0")
    with pytest.raises(IngestError, match="ALLOW_WEBCAM"):
        list(WebcamSampler(enabled=False).frames())
    with pytest.raises(RuntimeError, match="ALLOW_WEBCAM"):
        list(WebcamStub(enabled=True).frames())
    assert "No detections generated" in WEBCAM_REFUSED_MESSAGE


def test_webcam_allowed_uses_opener(monkeypatch):
    monkeypatch.setenv("ALLOW_WEBCAM", "1")

    class _Cap:
        def __init__(self):
            self.n = 0

        def isOpened(self):
            return True

        def read(self):
            self.n += 1
            if self.n > 3:
                return False, None
            return True, np.zeros((20, 20, 3), dtype=np.uint8)

        def release(self):
            return None

    frames = list(
        WebcamSampler(
            device=0,
            max_frames=3,
            enabled=True,
            opener=lambda _dev: _Cap(),
        ).frames()
    )
    assert len(frames) == 3


def test_missing_file_is_honest_error(stores):
    cameras, alerts, tmp = stores
    cam = cameras.create(
        name="Missing",
        source_type="file",
        uri="no/such/video.mp4",
        location_label="Vault",
    )
    pipe = CyberEyePipeline(
        store=alerts, adapter=MockVisionAdapter(), snapshot_dir=tmp / "snaps"
    )
    results = run_registered_cameras(pipe, [cam], cameras, project_root=tmp)
    assert results[0].error
    assert "not found" in results[0].error.lower()
    assert alerts.list_alerts() == []
    assert cameras.get(cam.id).last_error


def test_iter_unknown_source_type(stores):
    cameras, _alerts, tmp = stores
    cam = cameras.create(name="X", source_type="file", uri="MOCK")
    cam.source_type = "satellite"
    with pytest.raises(IngestError, match="Unknown camera source_type"):
        list(iter_camera_frames(cam, project_root=tmp))
