"""Open a registered camera and yield sampled frames.

File, RTSP, webcam, and MOCK synthetic frames share this entry point so
routes stay thin. Failures raise IngestError — never fake detections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

import numpy as np

from ingest.cameras import Camera, is_mock_uri, resolve_uri
from ingest.errors import IngestError
from ingest.rtsp import RtspSampler, rtsp_timeout_sec
from ingest.sampler import FrameSampler, SampledFrame
from ingest.webcam import WebcamSampler, parse_device


def iter_camera_frames(
    camera: Camera,
    *,
    max_frames: Optional[int] = None,
    project_root: str | Path | None = None,
    allow_webcam: Optional[bool] = None,
    timeout_sec: Optional[float] = None,
    rtsp_opener=None,
    webcam_opener=None,
) -> Generator[SampledFrame, None, None]:
    source_type = (camera.source_type or "").strip().lower()
    uri = resolve_uri(camera.uri)
    fps = camera.effective_fps()
    label = camera.source_label()
    root = Path(project_root) if project_root else Path.cwd()

    if source_type == "file":
        if is_mock_uri(uri) or is_mock_uri(camera.uri):
            # MOCK is a CI/demo stand-in, not a live stream — keep runs short.
            limit = 16 if max_frames is None else min(int(max_frames), 16)
            yield from mock_frames(
                max_frames=limit,
                sample_fps=fps,
                source_label=label,
            )
            return
        path = Path(uri)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise IngestError(
                f"Authorized video file not found: {path}. "
                "No detections generated."
            )
        yield from FrameSampler(
            path,
            sample_fps=fps,
            max_frames=max_frames,
            source_label=label,
        ).frames()
        return

    if source_type == "rtsp":
        yield from RtspSampler(
            uri,
            sample_fps=fps,
            max_frames=max_frames,
            source_label=label,
            timeout_sec=timeout_sec if timeout_sec is not None else rtsp_timeout_sec(),
            opener=rtsp_opener,
        ).frames()
        return

    if source_type == "webcam":
        yield from WebcamSampler(
            parse_device(uri or camera.uri or "0"),
            sample_fps=fps,
            max_frames=max_frames or 30,
            source_label=label,
            enabled=allow_webcam,
            opener=webcam_opener,
        ).frames()
        return

    raise IngestError(
        f"Unknown camera source_type {source_type!r}. No detections generated."
    )


def mock_frames(
    max_frames: int = 16,
    sample_fps: float = 2.0,
    source_label: str = "Synthetic sample",
) -> Generator[SampledFrame, None, None]:
    """Deterministic blank-pattern frames for CI and the seeded demo camera."""
    fps = max(0.1, float(sample_fps))
    for i in range(int(max_frames)):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 0] = (i * 17) % 255
        img[:, :, 1] = 40
        img[:, :, 2] = 60
        yield SampledFrame(
            index=i,
            timestamp_sec=i / fps,
            image_bgr=img,
            source_label=source_label,
        )
