"""Optional continuous recording while monitoring is on.

Off by default. Synthetic MOCK cameras are skipped. Incident clips are a
separate path under the clip directory.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ingest.archive import (
    DEFAULT_ARCHIVE_FPS,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_SEGMENT_SEC,
    ArchiveDiskFull,
    ArchiveError,
    ArchiveSkipped,
    ArchiveStore,
    SegmentWriter,
    adopt_finished_files,
    clamp_archive_fps,
    clamp_retention_days,
    clamp_segment_sec,
    discard_partial_segments,
    disk_cost_note,
    iter_capture_frames,
    open_archive_capture,
    prune_archive,
)
from ingest.cameras import is_mock_uri, resolve_uri

logger = logging.getLogger(__name__)

_LIVE_SOURCES = {"rtsp", "webcam"}


class ArchiveService:
    """Record enabled cameras into short segments while monitoring runs."""

    def __init__(self, app) -> None:
        self.app = app
        self.capture_opener = None
        self.rtsp_opener = None
        self.webcam_opener = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._workers: dict[str, threading.Thread] = {}
        self._worker_stops: dict[str, threading.Event] = {}
        self._file_done: dict[str, tuple] = {}
        self._retry_after: dict[str, float] = {}
        self._notes: dict[str, dict[str, str]] = {}
        self._enabled = False
        self._retention_days = DEFAULT_RETENTION_DAYS
        self._last_error = ""
        self._prune_counter = 0
        root = Path(app.config.get("ARCHIVE_DIR") or "data/archive")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.store = ArchiveStore(app.config.get("ARCHIVE_DB_PATH") or root.parent / "archive.db")
        discard_partial_segments(root)
        try:
            adopt_finished_files(self.store, root)
        except Exception:
            logger.exception("Could not index existing archive files")
        self._load_state()
        try:
            self.prune()
        except Exception:
            logger.exception("Archive prune failed during startup")

    def set_enabled(self, enabled: bool, actor: str = "operator") -> dict[str, Any]:
        with self._lock:
            self._enabled = bool(enabled)
            if self._enabled:
                self._notes.clear()
                self._last_error = ""
                self._retry_after.clear()
            self._persist()
        logger.info("Archive %s by %s", "on" if enabled else "off", actor)
        self.sync()
        return self.status()

    def set_retention(self, days: int, actor: str = "operator") -> dict[str, Any]:
        cleaned = clamp_retention_days(days, default=0)
        if cleaned <= 0:
            raise ValueError("Retention must be a whole number of days from 1 to 90.")
        with self._lock:
            self._retention_days = cleaned
            self._persist()
        logger.info("Archive retention set to %s day(s) by %s", cleaned, actor)
        self.prune()
        return self.status()

    def sync(self) -> dict[str, Any]:
        """Start or stop recorders to match archive + monitoring."""
        should = self._wants_recording()
        join_thread = None
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            if should and not alive:
                self._stop.clear()
                self._last_error = ""
                self._thread = threading.Thread(
                    target=self._loop,
                    name="mun-cyber-eye-archive",
                    daemon=True,
                )
                self._thread.start()
            elif not should:
                self._stop.set()
                for event in self._worker_stops.values():
                    event.set()
                join_thread = self._thread
        if (
            join_thread is not None
            and join_thread.is_alive()
            and join_thread is not threading.current_thread()
        ):
            join_thread.join(timeout=self._join_timeout())
            if join_thread.is_alive():
                self._set_last_error(
                    "Stop requested. The current archive segment is still finishing."
                )
        return self.status()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            for event in self._worker_stops.values():
                event.set()
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.1, float(timeout)))

    def prune(self) -> int:
        return prune_archive(self.store, self.root, self._retention_value())

    def status(self) -> dict[str, Any]:
        with self._lock:
            notes = {key: dict(value) for key, value in self._notes.items()}
            enabled = self._enabled
            retention = self._retention_days
            last_error = self._last_error
        fps = self._fps()
        segment_sec = self._segment_sec()
        errors = []
        skipped_mock = 0
        recording = False
        for camera_id, note in notes.items():
            state = note.get("state") or ""
            message = (note.get("message") or "").strip()
            if state == "recording":
                recording = True
            if state == "skipped":
                skipped_mock += 1
                continue
            if state == "error" and message:
                errors.append({"camera_id": camera_id, "message": message})
        notices = []
        if skipped_mock:
            notices.append("Synthetic MOCK cameras are not archived.")
        if enabled and not self._monitor_running():
            notices.append("Archive records when monitoring is on.")
        return {
            "enabled": enabled,
            "recording": recording,
            "retention_days": retention,
            "segment_sec": segment_sec,
            "fps": fps,
            "last_error": last_error,
            "errors": errors[:6],
            "notices": notices,
            "disk_note": disk_cost_note(fps),
            "segment_note": _segment_note(segment_sec),
        }

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._reconcile()
                    self._prune_counter += 1
                    if self._prune_counter >= 120:
                        self._prune_counter = 0
                        self.prune()
                except Exception as exc:
                    logger.exception("Archive loop failed")
                    self._set_last_error(str(exc) or exc.__class__.__name__)
                if self._stop.wait(0.5):
                    break
        finally:
            self._stop_workers()

    def _reconcile(self) -> None:
        if not self._wants_recording():
            self._stop_workers()
            return
        cameras = self._enabled_cameras()
        wanted: set[str] = set()
        for camera in cameras:
            if self._stop.is_set():
                break
            if not self._should_start(camera):
                continue
            wanted.add(camera.id)
            self._ensure_worker(camera.id)
        stale = [camera_id for camera_id in list(self._workers) if camera_id not in wanted]
        for camera_id in stale:
            self._stop_worker(camera_id)

    def _should_start(self, camera) -> bool:
        if time.time() < self._retry_after.get(camera.id, 0):
            return False
        source = (camera.source_type or "").strip().lower()
        if source == "file":
            signature = self._file_signature(camera)
            if signature[0] == "mock":
                self._note(camera.id, "skipped", "Synthetic MOCK is not archived.")
                return False
            if signature[0] == "missing":
                self._note(
                    camera.id,
                    "error",
                    "Authorized video file not found. No archive footage was recorded.",
                )
                return False
            if self._file_done.get(camera.id) == signature:
                self._note(camera.id, "idle", "")
                return False
        if source == "webcam" and not bool(self.app.config.get("ALLOW_WEBCAM")):
            self._note(
                camera.id,
                "error",
                "Webcam capture is disabled. No archive footage was recorded.",
            )
            return False
        if source == "rtsp":
            uri = resolve_uri(camera.uri)
            if not uri:
                self._note(
                    camera.id,
                    "error",
                    "RTSP URI is missing or the env secret is empty. "
                    "No archive footage was recorded.",
                )
                return False
        return True

    def _ensure_worker(self, camera_id: str) -> None:
        with self._lock:
            current = self._workers.get(camera_id)
            if current is not None and current.is_alive():
                return
            stop = threading.Event()
            self._worker_stops[camera_id] = stop
            thread = threading.Thread(
                target=self._worker_main,
                args=(camera_id,),
                name=f"mun-archive-{camera_id[:18]}",
                daemon=True,
            )
            self._workers[camera_id] = thread
            thread.start()

    def _worker_main(self, camera_id: str) -> None:
        try:
            self._run_camera(camera_id)
        except ArchiveDiskFull as exc:
            self._defer(camera_id, 30)
            self._note(camera_id, "error", str(exc))
            self._set_last_error(str(exc))
        except ArchiveSkipped as exc:
            self._note(camera_id, "skipped", str(exc))
        except ArchiveError as exc:
            self._defer(camera_id, 12)
            self._note(camera_id, "error", str(exc))
        except Exception as exc:
            logger.exception("Archive camera %s failed", camera_id)
            self._defer(camera_id, 12)
            message = str(exc) or exc.__class__.__name__
            self._note(camera_id, "error", message)
            self._set_last_error(message)
        finally:
            with self._lock:
                note = self._notes.get(camera_id)
                if note and note.get("state") == "recording":
                    self._notes[camera_id] = {"state": "idle", "message": ""}

    def _run_camera(self, camera_id: str) -> None:
        while not self._worker_should_stop(camera_id):
            camera = self._camera(camera_id)
            if camera is None or not camera.enabled:
                return
            source = (camera.source_type or "").strip().lower()
            signature = self._file_signature(camera) if source == "file" else None
            if signature is not None and signature[0] == "mock":
                raise ArchiveSkipped("Synthetic MOCK is not archived.")
            if signature is not None and self._file_done.get(camera_id) == signature:
                return
            self._note(camera_id, "recording", "")
            try:
                outcome = self._record_pass(camera)
            except ArchiveDiskFull:
                raise
            except ArchiveError as exc:
                self._note(camera_id, "error", str(exc))
                if source in _LIVE_SOURCES and not self._interruptible_wait(camera_id, 8):
                    continue
                raise
            if outcome == "stopped":
                return
            if source == "file":
                if outcome == "empty":
                    raise ArchiveError(
                        "Camera produced no frames. No archive footage was recorded."
                    )
                if signature is not None and signature[0] == "file":
                    self._file_done[camera_id] = signature
                self._note(camera_id, "idle", "")
                return
            if outcome == "empty":
                self._note(
                    camera_id,
                    "error",
                    "Stream produced no frames. No archive footage was recorded.",
                )
            else:
                self._note(
                    camera_id,
                    "error",
                    "Stream ended. Archive will try again. No footage was invented for the gap.",
                )
            if self._interruptible_wait(camera_id, 5.0):
                return

    def _record_pass(self, camera) -> str:
        cap = None
        writer: Optional[SegmentWriter] = None
        frames = 0
        try:
            cap = self._open_capture(camera)
            for frame, playback in iter_capture_frames(
                cap,
                self._fps(),
                lambda: self._worker_should_stop(camera.id),
            ):
                if writer is None:
                    writer = SegmentWriter(
                        self.root,
                        camera.id,
                        self.store,
                        fps=playback,
                        segment_sec=self._segment_sec(),
                        min_free_bytes=self._min_free_bytes(),
                    )
                writer.write(frame)
                frames += 1
            if self._worker_should_stop(camera.id):
                return "stopped"
            return "empty" if frames <= 0 else "eof"
        finally:
            if writer is not None:
                writer.close()
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    logger.debug("Archive capture release failed", exc_info=True)

    def _open_capture(self, camera):
        if self.capture_opener is not None:
            opened = self.capture_opener(camera)
            if opened is None:
                raise ArchiveError(
                    "Camera could not be opened. No archive footage was recorded."
                )
            return opened
        return open_archive_capture(
            camera,
            project_root=self.app.config.get("PROJECT_ROOT"),
            allow_webcam=bool(self.app.config.get("ALLOW_WEBCAM")),
            timeout_sec=float(self.app.config.get("RTSP_CONNECT_TIMEOUT_SEC") or 8),
            rtsp_opener=self.rtsp_opener,
            webcam_opener=self.webcam_opener,
        )

    def _stop_workers(self) -> None:
        with self._lock:
            ids = list(self._workers)
        for camera_id in ids:
            self._stop_worker(camera_id)

    def _stop_worker(self, camera_id: str) -> None:
        with self._lock:
            event = self._worker_stops.get(camera_id)
            thread = self._workers.get(camera_id)
            if event is not None:
                event.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            current = self._workers.get(camera_id)
            if current is thread:
                self._workers.pop(camera_id, None)
                self._worker_stops.pop(camera_id, None)

    def _worker_should_stop(self, camera_id: str) -> bool:
        if self._stop.is_set() or not self._wants_recording():
            return True
        event = self._worker_stops.get(camera_id)
        return event is not None and event.is_set()

    def _interruptible_wait(self, camera_id: str, seconds: float) -> bool:
        end = seconds
        waited = 0.0
        while waited < end:
            if self._worker_should_stop(camera_id):
                return True
            if self._stop.wait(0.2):
                return True
            waited += 0.2
        return self._worker_should_stop(camera_id)

    def _wants_recording(self) -> bool:
        with self._lock:
            enabled = self._enabled
        return bool(enabled) and self._monitor_running()

    def _monitor_running(self) -> bool:
        monitor = self.app.extensions.get("monitor")
        if monitor is None:
            return False
        try:
            return bool(monitor.status().get("running"))
        except Exception:
            return False

    def _enabled_cameras(self):
        cameras = self.app.extensions.get("camera_store")
        if cameras is None:
            return []
        return cameras.list_cameras(enabled_only=True)

    def _camera(self, camera_id: str):
        cameras = self.app.extensions.get("camera_store")
        if cameras is None:
            return None
        return cameras.get(camera_id)

    def _file_signature(self, camera) -> tuple:
        uri = resolve_uri(camera.uri)
        if is_mock_uri(uri) or is_mock_uri(camera.uri):
            return ("mock", camera.id)
        path = Path(uri)
        if not path.is_absolute():
            root = Path(self.app.config.get("PROJECT_ROOT") or ".")
            path = root / path
        if not path.exists():
            return ("missing", str(path))
        stat = path.stat()
        return ("file", str(path.resolve()), stat.st_mtime_ns, stat.st_size)

    def _fps(self) -> float:
        return clamp_archive_fps(self.app.config.get("ARCHIVE_FPS"), DEFAULT_ARCHIVE_FPS)

    def _segment_sec(self) -> int:
        return clamp_segment_sec(
            self.app.config.get("ARCHIVE_SEGMENT_SEC"), DEFAULT_SEGMENT_SEC
        )

    def _retention_value(self) -> int:
        with self._lock:
            return self._retention_days

    def _min_free_bytes(self) -> int:
        try:
            return max(0, int(self.app.config.get("ARCHIVE_MIN_FREE_BYTES") or 0))
        except (TypeError, ValueError):
            return 0

    def _join_timeout(self) -> float:
        try:
            return max(1.0, float(self.app.config.get("ARCHIVE_STOP_TIMEOUT_SEC") or 8))
        except (TypeError, ValueError):
            return 8.0

    def _defer(self, camera_id: str, seconds: float) -> None:
        self._retry_after[camera_id] = time.time() + max(0.5, float(seconds))

    def _note(self, camera_id: str, state: str, message: str) -> None:
        with self._lock:
            self._notes[camera_id] = {"state": state, "message": message or ""}

    def _set_last_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message or ""

    def _state_path(self) -> Path:
        return Path(self.app.config.get("ARCHIVE_STATE_PATH") or "data/archive_state.json")

    def _load_state(self) -> None:
        configured = clamp_retention_days(
            self.app.config.get("ARCHIVE_RETENTION_DAYS"),
            DEFAULT_RETENTION_DAYS,
        )
        enabled = bool(self.app.config.get("ARCHIVE_ENABLED"))
        path = self._state_path()
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                data = None
            if isinstance(data, dict):
                if "enabled" in data:
                    enabled = bool(data.get("enabled"))
                if data.get("retention_days") is not None:
                    configured = clamp_retention_days(
                        data.get("retention_days"), configured
                    )
        with self._lock:
            self._enabled = enabled
            self._retention_days = configured

    def _persist(self) -> None:
        path = self._state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "enabled": self._enabled,
                        "retention_days": self._retention_days,
                    }
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not save archive state: %s", exc)


def _segment_note(segment_sec: int) -> str:
    if segment_sec % 60 == 0:
        minutes = segment_sec // 60
        unit = "minute" if minutes == 1 else "minutes"
        return f"Segments are about {minutes} {unit}."
    return f"Segments are about {segment_sec} seconds."
