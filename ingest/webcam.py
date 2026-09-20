"""Webcam ingest, gated behind ALLOW_WEBCAM=1 (default off)."""

from __future__ import annotations

import logging
import os
from typing import Generator, Optional

import cv2

from ingest.errors import IngestError
from ingest.sampler import SampledFrame

logger = logging.getLogger(__name__)

WEBCAM_REFUSED_MESSAGE = (
    "Webcam capture is disabled. Set ALLOW_WEBCAM=1 only for an authorized "
    "local lab device. Production cameras must be registered as file or RTSP "
    "sources. No detections generated."
)


def allow_webcam(override: Optional[bool] = None) -> bool:
    if override is not None:
        return bool(override)
    return os.getenv("ALLOW_WEBCAM", "0").strip().lower() in {"1", "true", "yes", "on"}


def parse_device(uri: str) -> int | str:
    raw = (uri or "0").strip() or "0"
    if raw.isdigit():
        return int(raw)
    return raw


class WebcamSampler:
    """Sample frames from a local webcam after an explicit env opt-in."""

    def __init__(
        self,
        device: int | str = 0,
        sample_fps: float = 2.0,
        max_frames: Optional[int] = 30,
        source_label: str = "Authorized Webcam — Lab",
        enabled: Optional[bool] = None,
        opener=None,
    ) -> None:
        self.device = device
        self.sample_fps = max(0.1, float(sample_fps))
        self.max_frames = max_frames
        self.source_label = source_label
        self.enabled = allow_webcam(enabled)
        self.opener = opener or cv2.VideoCapture

    def frames(self) -> Generator[SampledFrame, None, None]:
        if not self.enabled:
            raise IngestError(WEBCAM_REFUSED_MESSAGE)

        cap = self.opener(self.device)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            raise IngestError(
                f"Webcam device {self.device} is unavailable. "
                "No detections generated."
            )

        try:
            limit = self.max_frames if self.max_frames is not None else 30
            for i in range(int(limit)):
                ok, frame = cap.read()
                if not ok or frame is None:
                    if i == 0:
                        raise IngestError(
                            f"Webcam device {self.device} produced no frames. "
                            "No detections generated."
                        )
                    break
                yield SampledFrame(
                    index=i,
                    timestamp_sec=i / self.sample_fps,
                    image_bgr=frame,
                    source_label=self.source_label,
                )
        finally:
            cap.release()
