"""Discover local cameras the console can register.

Webcam discovery probes OpenCV device indices and keeps the registry URI
convention: a decimal device index (``"0"``, ``"1"``, …). Network discovery
(RTSP/ONVIF) is not scanned here. ``discover_local_cameras`` is the place to
append another scanner later without changing the registry or the Cameras page.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

from ingest.cameras import is_env_ref, resolve_uri

logger = logging.getLogger(__name__)

DEFAULT_MAX_INDEX = 8
DEFAULT_MISS_STREAK = 3
DEFAULT_OPEN_TIMEOUT_SEC = 1.0
DEFAULT_SAMPLE_FPS = 2.0
MAX_DEVICE_INDEX = 64
_LINUX_DEVICE_PREFIX = "/dev/video"


@dataclass(frozen=True)
class DiscoveredCamera:
    """One device an operator can add from the detection results."""

    name: str
    source_type: str
    uri: str
    already_registered: bool = False


def webcam_index_from_uri(uri: str) -> Optional[int]:
    """Return a webcam device index stored the way the registry already does.

    Accepts ``"0"`` / ``" 1 "`` and an existing ``/dev/videoN`` path. Empty
    strings are not index 0. ``env:VAR`` is resolved first.
    """
    raw = uri or ""
    if is_env_ref(raw):
        raw = resolve_uri(raw)
    raw = (raw or "").strip()
    if not raw or is_env_ref(raw):
        return None
    if raw.isdigit():
        index = int(raw)
        if 0 <= index <= MAX_DEVICE_INDEX:
            return index
        return None
    if raw.startswith(_LINUX_DEVICE_PREFIX):
        tail = raw[len(_LINUX_DEVICE_PREFIX) :]
        if tail.isdigit():
            index = int(tail)
            if 0 <= index <= MAX_DEVICE_INDEX:
                return index
    return None


def registered_webcam_indices(cameras: Iterable[Any] | None) -> set[int]:
    """Device indices already stored as ``source_type=webcam``."""
    found: set[int] = set()
    for camera in cameras or []:
        source = (getattr(camera, "source_type", "") or "").strip().lower()
        if source != "webcam":
            continue
        index = webcam_index_from_uri(getattr(camera, "uri", "") or "")
        if index is not None:
            found.add(index)
    return found


def discover_webcams(
    existing: Iterable[Any] | None = None,
    *,
    opener: Callable[[int], Any] | None = None,
    max_index: int = DEFAULT_MAX_INDEX,
    miss_streak: int = DEFAULT_MISS_STREAK,
    grab_frame: bool = True,
    open_timeout_sec: float = DEFAULT_OPEN_TIMEOUT_SEC,
) -> list[DiscoveredCamera]:
    """Probe local webcam indices that open and can produce a frame.

    ``opener`` receives a device index and returns a capture, or ``None`` when
    that index is unavailable. The default opener uses OpenCV and gives up on
    an index that does not open within ``open_timeout_sec``. After
    ``miss_streak`` consecutive misses the scan stops so empty machines are
    not walked forever.
    """
    open_index = opener or (
        lambda index: _open_webcam_index(index, open_timeout_sec)
    )
    registered = registered_webcam_indices(existing)
    found: list[DiscoveredCamera] = []
    misses = 0
    streak = max(1, int(miss_streak))
    limit = max(0, min(int(max_index), MAX_DEVICE_INDEX + 1))
    for index in range(limit):
        usable = False
        try:
            usable = _capture_usable(open_index(index), grab_frame=grab_frame)
        except Exception:
            logger.debug("webcam index %s probe failed", index, exc_info=True)
            usable = False
        if not usable:
            misses += 1
            if misses >= streak:
                break
            continue
        misses = 0
        found.append(
            DiscoveredCamera(
                name=f"Webcam {index}",
                source_type="webcam",
                uri=str(index),
                already_registered=index in registered,
            )
        )
    return found


def discover_local_cameras(
    existing: Iterable[Any] | None = None,
    **kwargs: Any,
) -> list[DiscoveredCamera]:
    """Cameras on this machine that the console can offer to register.

    Webcam indices are included today. A future network scanner should return
    ``DiscoveredCamera`` rows and be concatenated here.
    """
    return discover_webcams(existing, **kwargs)


def add_discovered_webcams(
    store: Any,
    found: Sequence[DiscoveredCamera],
    *,
    only_uris: Optional[Iterable[str]] = None,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> list[Any]:
    """Register discovered webcams that are not already stored.

    ``only_uris`` limits the add to those device indices (the Add selected
    action). ``None`` adds every new webcam in ``found``. Matching is by
    webcam index, so ``"0"`` and ``/dev/video0`` do not create a second row.
    """
    limit: Optional[set[int]] = None
    if only_uris is not None:
        limit = set()
        for raw in only_uris:
            index = webcam_index_from_uri(str(raw))
            if index is not None:
                limit.add(index)
    registered = registered_webcam_indices(store.list_cameras())
    created: list[Any] = []
    seen: set[int] = set()
    for item in found:
        if (item.source_type or "").strip().lower() != "webcam":
            continue
        index = webcam_index_from_uri(item.uri)
        if index is None or index in seen or index in registered:
            continue
        if limit is not None and index not in limit:
            continue
        seen.add(index)
        created.append(
            store.create(
                name=(item.name or f"Webcam {index}").strip() or f"Webcam {index}",
                source_type="webcam",
                uri=str(index),
                enabled=True,
                sample_fps=sample_fps,
            )
        )
        registered.add(index)
    return created


def _capture_usable(cap: Any, *, grab_frame: bool) -> bool:
    if cap is None:
        return False
    try:
        is_opened = getattr(cap, "isOpened", None)
        if not callable(is_opened) or not is_opened():
            return False
        if not grab_frame:
            return True
        read = getattr(cap, "read", None)
        if not callable(read):
            return False
        ok, frame = read()
        return bool(ok) and frame is not None
    except Exception:
        logger.debug("webcam probe read failed", exc_info=True)
        return False
    finally:
        _release(cap)


def _release(cap: Any) -> None:
    if cap is None:
        return
    release = getattr(cap, "release", None)
    if not callable(release):
        return
    try:
        release()
    except Exception:
        logger.debug("webcam release failed", exc_info=True)


def _call_with_timeout(fn: Callable[[], Any], timeout_sec: float) -> Any:
    """Return ``fn()`` or ``None`` when it does not finish in time."""
    state: dict[str, Any] = {"value": None, "abandon": False}
    lock = threading.Lock()

    def _run() -> None:
        value = None
        try:
            value = fn()
        except Exception:
            logger.debug("webcam open failed", exc_info=True)
            return
        with lock:
            if state["abandon"]:
                _release(value)
                return
            state["value"] = value

    thread = threading.Thread(target=_run, daemon=True, name="webcam-probe")
    thread.start()
    thread.join(max(0.05, float(timeout_sec)))
    if thread.is_alive():
        with lock:
            state["abandon"] = True
            value = state["value"]
            state["value"] = None
        _release(value)
        return None
    return state["value"]


def _preferred_backend(cv2_module: Any) -> int:
    if sys.platform == "win32":
        name = "CAP_DSHOW"
    elif sys.platform == "darwin":
        name = "CAP_AVFOUNDATION"
    else:
        name = "CAP_V4L2"
    return int(getattr(cv2_module, name, 0) or 0)


def _open_webcam_index(index: int, timeout_sec: float) -> Any:
    """Open one device index, or return None if it fails or stalls."""

    def _open() -> Any:
        import cv2

        cap = None
        try:
            cap = cv2.VideoCapture()
            timeout_ms = int(max(float(timeout_sec), 0.2) * 1000)
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
            backend = _preferred_backend(cv2)
            opened = cap.open(int(index), backend) if backend else cap.open(int(index))
            if not opened or not cap.isOpened():
                cap.release()
                return None
            owned = cap
            cap = None
            return owned
        except Exception:
            logger.debug("webcam index %s open failed", index, exc_info=True)
            return None
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    logger.debug("webcam release failed", exc_info=True)

    return _call_with_timeout(_open, timeout_sec)
