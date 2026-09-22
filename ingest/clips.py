"""Short incident clips from the sampled feed around a suspected event.

Default window is 5 seconds before the detection and 5 seconds after
(about 10 seconds, hard cap 15). Frames are the ones the pipeline already
sampled — not a separate upload. Oldest clip files can be pruned from disk;
alert rows are never deleted here.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from ingest.sampler import SampledFrame

logger = logging.getLogger(__name__)

DEFAULT_CLIP_PRE_SEC = 5.0
DEFAULT_CLIP_POST_SEC = 5.0
DEFAULT_CLIP_MAX_SEC = 15.0
DEFAULT_MAX_INCIDENT_CLIPS = 400

_VIDEO_SUFFIXES = {".mp4", ".avi", ".webm"}


def safe_token(value: str, fallback: str = "cam") -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "-_" else "-" for ch in (value or "")
    )
    return (cleaned[:24] or fallback)


def playback_fps(frames: Sequence[SampledFrame]) -> float:
    """Play sampled frames across the same span they were captured."""
    if len(frames) < 2:
        return 8.0
    span = float(frames[-1].timestamp_sec) - float(frames[0].timestamp_sec)
    if span <= 0.05:
        return 8.0
    fps = (len(frames) - 1) / span
    return float(max(1.0, min(15.0, fps)))


def clip_duration_sec(frames: Sequence[SampledFrame]) -> float:
    if len(frames) < 2:
        return 0.0
    return max(0.0, float(frames[-1].timestamp_sec) - float(frames[0].timestamp_sec))


def select_clip_frames(
    frames: Iterable[SampledFrame],
    *,
    event_ts: float,
    end_ts: float,
    pre_sec: float,
) -> list[SampledFrame]:
    start = float(event_ts) - max(0.0, float(pre_sec))
    end = max(float(end_ts), float(event_ts))
    selected = [
        frame
        for frame in frames
        if start - 1e-3 <= float(frame.timestamp_sec) <= end + 1e-3
    ]
    return selected


def prune_incident_clips(directory: str | Path, max_clips: int) -> int:
    """Drop oldest clip files when over the cap. Does not touch alert rows."""
    if max_clips <= 0:
        return 0
    root = Path(directory)
    if not root.is_dir():
        return 0
    files = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES
    ]
    files.sort(key=lambda path: path.stat().st_mtime)
    extra = len(files) - int(max_clips)
    removed = 0
    for path in files[: max(0, extra)]:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("Could not prune clip %s: %s", path.name, exc)
    return removed


def write_incident_clip(
    frames: Sequence[SampledFrame],
    directory: str | Path,
    *,
    camera_id: str = "",
    category: str = "incident",
) -> Optional[Path]:
    """Write a short clip. Prefer H.264 MP4 when ffmpeg is available."""
    if not frames:
        return None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    token = safe_token(camera_id)
    cat = safe_token(category, fallback="incident")
    stem = f"{token}_{uuid.uuid4().hex[:12]}_{cat}"
    images = [_frame_image(frame) for frame in frames]
    images = [img for img in images if img is not None]
    if not images:
        return None
    if len(images) == 1:
        images = [images[0], images[0].copy()]
    height, width = images[0].shape[:2]
    width = _even(width)
    height = _even(height)
    prepared = [_fit(img, width, height) for img in images]
    fps = playback_fps(frames if len(frames) >= 2 else [frames[0], frames[0]])

    avi_path = root / f"{stem}.avi"
    if not _write_mjpg(avi_path, prepared, fps, width, height):
        mp4_path = root / f"{stem}.mp4"
        if _write_mp4v(mp4_path, prepared, fps, width, height):
            return mp4_path
        return None

    mp4_path = root / f"{stem}.mp4"
    if _transcode_h264(avi_path, mp4_path):
        try:
            avi_path.unlink()
        except OSError:
            pass
        return mp4_path
    return avi_path


def _frame_image(frame: SampledFrame) -> Optional[np.ndarray]:
    image = getattr(frame, "image_bgr", None)
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim < 2:
        return None
    return np.ascontiguousarray(image)


def _even(value: int) -> int:
    size = int(value)
    if size <= 1:
        return 2
    return size if size % 2 == 0 else size - 1


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(image)


def _write_mjpg(
    path: Path,
    frames: Sequence[np.ndarray],
    fps: float,
    width: int,
    height: int,
) -> bool:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path.is_file() and path.stat().st_size > 0


def _write_mp4v(
    path: Path,
    frames: Sequence[np.ndarray],
    fps: float,
    width: int,
    height: int,
) -> bool:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path.is_file() and path.stat().st_size > 0


def transcode_h264(src: Path, dest: Path, *, timeout: float = 60) -> bool:
    """Convert a clip to H.264 MP4 when ffmpeg is installed."""
    return _transcode_h264(src, dest, timeout=timeout)


def _transcode_h264(src: Path, dest: Path, timeout: float = 60) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            check=False,
            timeout=timeout,
            capture_output=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg clip transcode failed: %s", exc)
        return False
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size <= 0:
        logger.warning(
            "ffmpeg clip transcode failed: %s",
            (result.stderr or b"").decode("utf-8", errors="replace")[:300],
        )
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return False
    return True
