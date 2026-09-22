"""Background watch of enabled, authorized cameras.

Monitoring runs inside the console process. Recorded files and MOCK URIs are
reviewed once per session (again if the file changes). RTSP and webcam feeds
are sampled continuously until Stop. Suspected incidents use the pipeline clip
path. Humans still acknowledge, dismiss, escalate, or reopen.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ingest.cameras import is_mock_uri, resolve_uri
from ingest.clips import (
    DEFAULT_CLIP_MAX_SEC,
    DEFAULT_CLIP_POST_SEC,
    DEFAULT_CLIP_PRE_SEC,
    DEFAULT_MAX_INCIDENT_CLIPS,
)
from pipeline import CyberEyePipeline, run_registered_cameras

logger = logging.getLogger(__name__)

_LIVE_SOURCES = {"rtsp", "webcam"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _werkzeug_parent() -> bool:
    """The Flask debug reloader parent must not start a second worker."""
    debug = os.getenv("FLASK_DEBUG", "0").strip() == "1"
    return debug and os.getenv("WERKZEUG_RUN_MAIN") != "true"


class MonitorService:
    """Start/stop continuous monitoring tied to the Flask process."""

    def __init__(self, app) -> None:
        self.app = app
        self.rtsp_opener = None
        self.webcam_opener = None
        self._lock = threading.Lock()
        self._cycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pipeline: Optional[CyberEyePipeline] = None
        self._built_signature: Optional[tuple] = None
        self._seen_files: dict[str, tuple] = {}
        self._desired = "off"
        self._status: dict[str, Any] = {
            "running": False,
            "desired": "off",
            "started_at": "",
            "last_cycle_at": "",
            "cycles": 0,
            "alerts_created": 0,
            "last_error": "",
        }

    def restore(self) -> dict[str, Any]:
        """Apply the saved on/off choice, or MONITOR_AUTOSTART when unset."""
        if _werkzeug_parent():
            return self.status()
        desired = self._read_desired()
        if desired is None:
            desired = "on" if self.app.config.get("MONITOR_AUTOSTART") else "off"
        if desired == "on":
            return self.start(actor="system")
        self._desired = "off"
        with self._lock:
            self._status["desired"] = "off"
        return self.status()

    def start(self, actor: str = "operator") -> dict[str, Any]:
        with self._lock:
            self._desired = "on"
            self._status["desired"] = "on"
            self._persist()
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._seen_files.clear()
                self._pipeline = None
                self._built_signature = None
                self._status["started_at"] = _utc_now()
                self._status["alerts_created"] = 0
                self._status["cycles"] = 0
                self._status["last_error"] = ""
                self._thread = threading.Thread(
                    target=self._loop,
                    name="mun-cyber-eye-monitor",
                    daemon=True,
                )
                self._thread.start()
                logger.info("Continuous monitoring started by %s", actor)
            status = self._status_unlocked()
        self._sync_archive()
        return status

    def stop(self, actor: str = "operator", join_timeout: float = 45.0) -> dict[str, Any]:
        with self._lock:
            self._desired = "off"
            self._status["desired"] = "off"
            self._persist()
            thread = self._thread
        self._stop.set()
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=max(0.1, float(join_timeout)))
        with self._lock:
            if thread is not None and thread.is_alive():
                self._status["last_error"] = (
                    "Stop requested. The current camera pass is still finishing."
                )
            else:
                self._thread = None
                self._status["running"] = False
            logger.info("Continuous monitoring stop requested by %s", actor)
            status = self._status_unlocked()
        self._sync_archive()
        return status

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def run_once(self) -> dict[str, Any]:
        """One sweep of enabled cameras. The background loop and tests call this."""
        with self._cycle_lock:
            return self._run_once_locked()

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception as exc:
                    logger.exception("Monitor sweep failed")
                    with self._lock:
                        self._status["last_error"] = str(exc) or exc.__class__.__name__
                if self._stop.wait(self._pause_sec()):
                    break
        finally:
            with self._lock:
                self._status["running"] = False

    def _run_once_locked(self) -> dict[str, Any]:
        cameras = self.app.extensions["camera_store"]
        enabled = cameras.list_cameras(enabled_only=True)
        alerts_created = 0
        frames = 0
        failures = 0
        last_error = ""
        observed_backend = ""
        pipeline = self._ensure_pipeline()
        for camera in enabled:
            if self._stop.is_set():
                break
            if self._file_already_reviewed(camera):
                continue
            results = run_registered_cameras(
                pipeline,
                [camera],
                cameras,
                max_frames=self._max_frames_for(camera),
                project_root=self.app.config.get("PROJECT_ROOT"),
                allow_webcam=bool(self.app.config.get("ALLOW_WEBCAM")),
                timeout_sec=float(self.app.config.get("RTSP_CONNECT_TIMEOUT_SEC") or 8),
                rtsp_opener=self.rtsp_opener,
                webcam_opener=self.webcam_opener,
                stop_check=self._stop_check_for(camera),
            )
            for result in results:
                frames += result.frames_processed
                alerts_created += len(result.alerts_created)
                if result.frames_processed and getattr(result, "objects_backend", ""):
                    observed_backend = str(result.objects_backend)
                if result.error:
                    failures += 1
                    last_error = result.error
                elif (camera.source_type or "").lower() == "file":
                    self._seen_files[camera.id] = self._file_signature(camera)
        with self._lock:
            self._status["cycles"] = int(self._status.get("cycles") or 0) + 1
            self._status["last_cycle_at"] = _utc_now()
            self._status["alerts_created"] = (
                int(self._status.get("alerts_created") or 0) + alerts_created
            )
            self._status["last_error"] = last_error
            if observed_backend:
                self._status["objects_backend"] = observed_backend
        log = logger.info if (frames or alerts_created or failures) else logger.debug
        log(
            "Monitor sweep cameras=%s frames=%s alerts=%s failures=%s",
            len(enabled),
            frames,
            alerts_created,
            failures,
        )
        return {
            "cameras": len(enabled),
            "frames": frames,
            "alerts": alerts_created,
            "failures": failures,
            "last_error": last_error,
        }

    def _stop_check_for(self, camera):
        if (camera.source_type or "").lower() not in _LIVE_SOURCES:
            return None
        limit = self._pass_limit()
        allowance = self._post_allowance(camera)

        def stop_check(frames_seen: int, incident_open: bool) -> bool:
            if self._stop.is_set() and not incident_open:
                return True
            if frames_seen >= limit and not incident_open:
                return True
            if frames_seen >= limit + allowance:
                return True
            return False

        return stop_check

    def _max_frames_for(self, camera) -> Optional[int]:
        source = (camera.source_type or "").lower()
        if source in _LIVE_SOURCES:
            return self._pass_limit() + self._post_allowance(camera)
        try:
            return int(self.app.config.get("MAX_FRAMES_PER_RUN") or 120)
        except (TypeError, ValueError):
            return 120

    def _pass_limit(self) -> int:
        try:
            return max(1, int(self.app.config.get("MONITOR_MAX_FRAMES_PER_PASS") or 24))
        except (TypeError, ValueError):
            return 24

    def _post_allowance(self, camera) -> int:
        try:
            post = float(self.app.config.get("CLIP_POST_SEC") or DEFAULT_CLIP_POST_SEC)
        except (TypeError, ValueError):
            post = DEFAULT_CLIP_POST_SEC
        fps = 2.0
        try:
            fps = float(camera.effective_fps())
        except (TypeError, ValueError, AttributeError):
            fps = 2.0
        return max(1, int(post * max(0.1, fps)) + 2)

    def _pause_sec(self) -> float:
        try:
            return max(0.05, float(self.app.config.get("MONITOR_CYCLE_PAUSE_SEC") or 0.5))
        except (TypeError, ValueError):
            return 0.5

    def _active_checkpoint_path(self) -> str:
        """Active pointer on disk, else the configured checkpoint. Same rule as Run."""
        from vision.checkpoint_config import resolve_checkpoint_path

        cfg = self.app.config
        return str(
            resolve_checkpoint_path(
                root=cfg.get("PROJECT_ROOT") or ".",
                active_file=cfg.get("ACTIVE_CHECKPOINT_FILE"),
                env_fallback=str(cfg.get("ACTIVITY_CHECKPOINT") or ""),
            )
        )

    def _object_mode(self) -> str:
        raw = self.app.config.get("OBJECTS_BACKEND")
        if raw is None or str(raw).strip() == "":
            raw = os.getenv("OBJECTS_BACKEND", "auto")
        mode = str(raw).strip().lower() or "auto"
        return mode

    def _runtime_signature(self) -> tuple:
        path = self._active_checkpoint_path()
        mtime = 0
        candidate = Path(path)
        if candidate.is_file():
            try:
                mtime = candidate.stat().st_mtime_ns
            except OSError:
                mtime = 0
        backend = str(self.app.config.get("VISION_BACKEND") or "auto")
        return (path, mtime, self._object_mode(), backend)

    def _ensure_pipeline(self) -> CyberEyePipeline:
        signature = self._runtime_signature()
        if self._pipeline is None or self._built_signature != signature:
            # Keep one-shot Run and monitoring on the same checkpoint.
            self.app.config["ACTIVITY_CHECKPOINT"] = signature[0]
            self._pipeline = self._build_pipeline()
            self._built_signature = signature
        return self._pipeline

    def _build_pipeline(self) -> CyberEyePipeline:
        from vision.detector import create_adapter

        cfg = self.app.config
        adapter = create_adapter(
            cfg.get("VISION_BACKEND"),
            checkpoint=cfg.get("ACTIVITY_CHECKPOINT"),
        )
        try:
            pre = float(cfg.get("CLIP_PRE_SEC") or DEFAULT_CLIP_PRE_SEC)
        except (TypeError, ValueError):
            pre = DEFAULT_CLIP_PRE_SEC
        try:
            post = float(cfg.get("CLIP_POST_SEC") or DEFAULT_CLIP_POST_SEC)
        except (TypeError, ValueError):
            post = DEFAULT_CLIP_POST_SEC
        try:
            cap = float(cfg.get("CLIP_MAX_SEC") or DEFAULT_CLIP_MAX_SEC)
        except (TypeError, ValueError):
            cap = DEFAULT_CLIP_MAX_SEC
        try:
            max_clips = int(cfg.get("MAX_INCIDENT_CLIPS") or DEFAULT_MAX_INCIDENT_CLIPS)
        except (TypeError, ValueError):
            max_clips = DEFAULT_MAX_INCIDENT_CLIPS
        try:
            cooldown = float(cfg.get("MONITOR_ALERT_COOLDOWN_SEC") or 60)
        except (TypeError, ValueError):
            cooldown = 60.0
        return CyberEyePipeline(
            store=self.app.extensions["alert_store"],
            adapter=adapter,
            snapshot_dir=cfg.get("SNAPSHOT_DIR") or "data/snapshots",
            notifier=self.app.extensions.get("notifier"),
            activity_data_root=cfg.get("ACTIVITY_DATA_ROOT"),
            object_mode=self._object_mode(),
            clip_dir=cfg.get("CLIP_DIR"),
            clip_pre_sec=pre,
            clip_post_sec=post,
            clip_max_sec=cap,
            max_incident_clips=max_clips,
            alert_cooldown_sec=cooldown,
        )

    def _file_already_reviewed(self, camera) -> bool:
        if (camera.source_type or "").lower() != "file":
            return False
        signature = self._file_signature(camera)
        if signature[0] == "missing":
            return False
        return self._seen_files.get(camera.id) == signature

    def _file_signature(self, camera) -> tuple:
        uri = resolve_uri(camera.uri)
        if is_mock_uri(uri) or is_mock_uri(camera.uri):
            return ("mock", camera.id, (camera.uri or "").strip())
        path = Path(uri)
        if not path.is_absolute():
            root = Path(self.app.config.get("PROJECT_ROOT") or ".")
            path = root / path
        if not path.exists():
            return ("missing", str(path))
        stat = path.stat()
        return ("file", str(path.resolve()), stat.st_mtime_ns, stat.st_size)

    def _status_unlocked(self) -> dict[str, Any]:
        data = dict(self._status)
        alive = self._thread is not None and self._thread.is_alive()
        data["running"] = alive
        data["desired"] = self._desired
        checkpoint = ""
        object_mode = "auto"
        try:
            checkpoint = self._active_checkpoint_path()
            object_mode = self._object_mode()
        except Exception:
            checkpoint = str(self.app.config.get("ACTIVITY_CHECKPOINT") or "")
        data["activity_checkpoint"] = checkpoint
        data["activity_checkpoint_name"] = Path(checkpoint).name if checkpoint else ""
        data["objects_mode"] = object_mode
        observed = str(self._status.get("objects_backend") or "")
        if observed:
            data["objects_backend"] = observed
        elif object_mode in {"off", "none", "disabled"}:
            data["objects_backend"] = "off"
        else:
            data["objects_backend"] = "unavailable"
        self._status["running"] = alive
        self._status["desired"] = self._desired
        return data

    def _sync_archive(self) -> None:
        """Archive records only while monitoring is running."""
        try:
            archive = self.app.extensions.get("archive")
        except Exception:
            return
        if archive is None:
            return
        try:
            archive.sync()
        except Exception:
            logger.exception("Archive sync failed")

    def _state_path(self) -> Path:
        return Path(self.app.config.get("MONITOR_STATE_PATH") or "data/monitor_state.json")

    def _read_desired(self) -> Optional[str]:
        path = self._state_path()
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        desired = str(data.get("desired") or "").strip().lower()
        if desired in {"on", "off"}:
            return desired
        return None

    def _persist(self) -> None:
        path = self._state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"desired": self._desired}),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not save monitoring state: %s", exc)
