"""RTSP open via OpenCV VideoCapture with an honest timeout.

Failures raise IngestError. Callers must record last_error and must not
emit fake detections.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Generator, Optional

import cv2
import numpy as np

from ingest.errors import IngestError
from ingest.sampler import SampledFrame

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SEC = 8.0


def rtsp_timeout_sec(override: Optional[float] = None) -> float:
    if override is not None:
        return max(1.0, float(override))
    raw = os.getenv("RTSP_CONNECT_TIMEOUT_SEC", str(int(DEFAULT_TIMEOUT_SEC)))
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SEC


class RtspSampler:
    """Sample frames from an authorized RTSP URL."""

    def __init__(
        self,
        uri: str,
        sample_fps: float = 2.0,
        max_frames: Optional[int] = None,
        source_label: str = "Authorized Camera — RTSP",
        timeout_sec: Optional[float] = None,
        opener=None,
    ) -> None:
        self.uri = (uri or "").strip()
        self.sample_fps = max(0.1, float(sample_fps))
        self.max_frames = max_frames
        self.source_label = source_label
        self.timeout_sec = rtsp_timeout_sec(timeout_sec)
        self.opener = opener or _open_rtsp

    def frames(self) -> Generator[SampledFrame, None, None]:
        if not self.uri:
            raise IngestError(
                "RTSP URI is missing or the env secret is empty. "
                "No detections generated."
            )
        cap = self.opener(self.uri, self.timeout_sec)
        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if native_fps <= 0:
                native_fps = 25.0
            interval = max(1, int(round(native_fps / self.sample_fps)))
            frame_idx = 0
            emitted = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if emitted == 0:
                        raise IngestError(
                            "RTSP stream opened but produced no frames "
                            "(offline or unauthorized credentials). "
                            "No detections generated."
                        )
                    break
                if not isinstance(frame, np.ndarray):
                    continue
                if frame_idx % interval == 0:
                    yield SampledFrame(
                        index=emitted,
                        timestamp_sec=frame_idx / native_fps,
                        image_bgr=frame,
                        source_label=self.source_label,
                    )
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        break
                frame_idx += 1
            logger.info("Sampled %s RTSP frames from authorized feed", emitted)
        finally:
            cap.release()


def _open_rtsp(uri: str, timeout_sec: float):
    """Open RTSP with OpenCV; abort if the connect exceeds timeout_sec."""
    timeout_ms = int(timeout_sec * 1000)
    holder: dict = {"cap": None, "error": None}

    def _connect() -> None:
        try:
            cap = cv2.VideoCapture()
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
            backend = getattr(cv2, "CAP_FFMPEG", 0)
            opened = cap.open(uri, backend) if backend else cap.open(uri)
            if not opened or not cap.isOpened():
                cap.release()
                holder["error"] = IngestError(
                    "Unable to open authorized RTSP feed "
                    f"(timeout {timeout_sec:.0f}s or camera offline). "
                    "No detections generated."
                )
                return
            holder["cap"] = cap
        except IngestError as exc:
            holder["error"] = exc
        except Exception as exc:  # pragma: no cover - OpenCV backend variance
            holder["error"] = IngestError(
                f"RTSP open failed: {exc}. No detections generated."
            )

    thread = threading.Thread(target=_connect, daemon=True, name="rtsp-open")
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise IngestError(
            f"RTSP connect timed out after {timeout_sec:.0f}s. "
            "Camera is treated as offline. No detections generated."
        )
    if holder["error"] is not None:
        raise holder["error"]
    cap = holder["cap"]
    if cap is None:
        raise IngestError(
            "Unable to open authorized RTSP feed. No detections generated."
        )
    return cap
