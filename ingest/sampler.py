"""Frame sampler for authorized video files and optional webcam stub.

Only authorized / operator-supplied sources should be used. This prototype
does not connect to live production cameras by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    """A single sampled frame from an authorized source."""

    index: int
    timestamp_sec: float
    image_bgr: np.ndarray
    source_label: str


class FrameSampler:
    """Sample frames from a video file at a target FPS."""

    def __init__(
        self,
        source_path: str | Path,
        sample_fps: float = 2.0,
        max_frames: Optional[int] = None,
        source_label: str = "Authorized Camera — File",
    ) -> None:
        self.source_path = Path(source_path)
        self.sample_fps = max(0.1, float(sample_fps))
        self.max_frames = max_frames
        self.source_label = source_label

    def frames(self) -> Generator[SampledFrame, None, None]:
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Authorized video file not found: {self.source_path}"
            )

        cap = cv2.VideoCapture(str(self.source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.source_path}")

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if native_fps <= 0:
                native_fps = 30.0
            interval = max(1, int(round(native_fps / self.sample_fps)))
            frame_idx = 0
            emitted = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % interval == 0:
                    ts = frame_idx / native_fps
                    yield SampledFrame(
                        index=emitted,
                        timestamp_sec=ts,
                        image_bgr=frame,
                        source_label=self.source_label,
                    )
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        break
                frame_idx += 1

            logger.info(
                "Sampled %s frames from %s (native_fps=%.2f, interval=%s)",
                emitted,
                self.source_path.name,
                native_fps,
                interval,
            )
        finally:
            cap.release()

    def __iter__(self) -> Iterator[SampledFrame]:
        return self.frames()


class WebcamStub:
    """Optional webcam stub for local demos.

    Disabled by default. Operators must explicitly enable webcam capture.
    Frames are treated as authorized local lab sources only — never as
    production surveillance without proper authorization.
    """

    def __init__(
        self,
        device_index: int = 0,
        sample_fps: float = 2.0,
        max_frames: int = 30,
        source_label: str = "Authorized Webcam — Lab Stub",
        enabled: bool = False,
    ) -> None:
        self.device_index = device_index
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.source_label = source_label
        self.enabled = enabled

    def frames(self) -> Generator[SampledFrame, None, None]:
        if not self.enabled:
            raise RuntimeError(
                "WebcamStub is disabled. Pass enabled=True only for authorized "
                "local lab demos. Production cameras require proper authorization."
            )

        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Webcam device {self.device_index} unavailable. "
                "Use video file ingest instead."
            )

        try:
            delay_ms = int(1000 / max(0.1, self.sample_fps))
            for i in range(self.max_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                yield SampledFrame(
                    index=i,
                    timestamp_sec=i / self.sample_fps,
                    image_bgr=frame,
                    source_label=self.source_label,
                )
                # Brief pause to approximate target FPS without busy-looping
                if delay_ms > 0:
                    cv2.waitKey(1)
        finally:
            cap.release()
