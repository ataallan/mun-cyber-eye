"""Continuous archive segments for authorized cameras.

Incident clips stay in ``data/clips/``. This module stores longer recordings
under ``data/archive/<camera_id>/`` and deletes segments older than the
retention window. It does not delete alert rows.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from ingest.cameras import is_mock_uri, resolve_uri
from ingest.clips import safe_token, transcode_h264
from ingest.errors import IngestError
from ingest.rtsp import _open_rtsp
from ingest.webcam import WEBCAM_REFUSED_MESSAGE, parse_device

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 1
DEFAULT_SEGMENT_SEC = 120
DEFAULT_ARCHIVE_FPS = 8.0
MIN_RETENTION_DAYS = 1
MAX_RETENTION_DAYS = 90
MIN_SEGMENT_SEC = 60
MAX_SEGMENT_SEC = 300
MIN_ARCHIVE_FPS = 1.0
MAX_ARCHIVE_FPS = 15.0

_VIDEO_SUFFIXES = {".mp4", ".avi", ".webm"}
_STAMP_RE = re.compile(r"^(\d{8}T\d{6}Z)")
_ISO = "%Y-%m-%dT%H:%M:%SZ"


class ArchiveError(Exception):
    """Archive could not record. Callers must not invent footage."""


class ArchiveSkipped(ArchiveError):
    """This camera is not an archive source (for example synthetic MOCK)."""


class ArchiveDiskFull(ArchiveError):
    """Free space is below the guard. No new segment is started."""


@dataclass
class ArchiveSegment:
    id: str
    camera_id: str
    relpath: str
    started_at: str
    ended_at: str
    nbytes: int
    playable: bool = True

    @property
    def suffix(self) -> str:
        return Path(self.relpath).suffix.lower()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime(_ISO)


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, _ISO).replace(tzinfo=timezone.utc)


def clamp_retention_days(value, default: int = DEFAULT_RETENTION_DAYS) -> int:
    try:
        days = int(value)
    except (TypeError, ValueError):
        return default
    if days < MIN_RETENTION_DAYS or days > MAX_RETENTION_DAYS:
        return default
    return days


def clamp_segment_sec(value, default: int = DEFAULT_SEGMENT_SEC) -> int:
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(MIN_SEGMENT_SEC, min(MAX_SEGMENT_SEC, seconds))


def clamp_archive_fps(value, default: float = DEFAULT_ARCHIVE_FPS) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return default
    if fps <= 0:
        return default
    return max(MIN_ARCHIVE_FPS, min(MAX_ARCHIVE_FPS, fps))


def disk_cost_note(fps: float) -> str:
    """Honest planning range at the archive frame rate, before H.264."""
    rate = clamp_archive_fps(fps)
    low = rate * 12_000 * 3600 / (1024 ** 3)
    high = rate * 40_000 * 3600 / (1024 ** 3)
    return f"About {low:.1f}–{high:.1f} GB per camera per hour."


def archive_window_bounds(choice: str, now: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """``today``, ``1``, ``5``, or another day count."""
    moment = now or utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    label = (choice or "").strip().lower()
    if label == "today":
        local = moment.astimezone()
        start = local.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.astimezone(timezone.utc), moment
    try:
        days = int(label)
    except (TypeError, ValueError):
        days = 1
    days = max(1, min(MAX_RETENTION_DAYS, days))
    return moment - timedelta(days=days), moment


class ArchiveStore:
    """Light index of finished segments. The files remain the footage."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS archive_segments (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    relpath TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    nbytes INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_archive_camera_time
                    ON archive_segments(camera_id, started_at DESC);
                """
            )

    def add_segment(
        self,
        *,
        camera_id: str,
        relpath: str,
        started_at: str,
        ended_at: str,
        nbytes: int,
        segment_id: Optional[str] = None,
    ) -> ArchiveSegment:
        row_id = segment_id or uuid.uuid4().hex
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO archive_segments (
                    id, camera_id, relpath, started_at, ended_at, nbytes
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (row_id, camera_id, relpath, started_at, ended_at, int(nbytes)),
            )
        return ArchiveSegment(
            id=row_id,
            camera_id=camera_id,
            relpath=relpath,
            started_at=started_at,
            ended_at=ended_at,
            nbytes=int(nbytes),
        )

    def get(self, segment_id: str) -> Optional[ArchiveSegment]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM archive_segments WHERE id = ?",
                (segment_id,),
            ).fetchone()
        return _row_to_segment(row) if row else None

    def known_relpaths(self) -> set[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT relpath FROM archive_segments").fetchall()
        return {str(row["relpath"]) for row in rows}

    def list_segments(
        self,
        *,
        camera_id: str = "",
        start: str = "",
        end: str = "",
    ) -> list[ArchiveSegment]:
        clauses = ["1 = 1"]
        params: list[str] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if start:
            clauses.append("ended_at >= ?")
            params.append(start)
        if end:
            clauses.append("started_at <= ?")
            params.append(end)
        sql = (
            "SELECT * FROM archive_segments WHERE "
            + " AND ".join(clauses)
            + " ORDER BY started_at DESC"
        )
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_segment(row) for row in rows]

    def segments_ended_before(self, cutoff: str) -> list[ArchiveSegment]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM archive_segments
                WHERE ended_at != '' AND ended_at < ?
                ORDER BY ended_at ASC
                """,
                (cutoff,),
            ).fetchall()
        return [_row_to_segment(row) for row in rows]

    def delete_segment(self, segment_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM archive_segments WHERE id = ?", (segment_id,))


def _row_to_segment(row: sqlite3.Row) -> ArchiveSegment:
    return ArchiveSegment(
        id=row["id"],
        camera_id=row["camera_id"],
        relpath=row["relpath"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        nbytes=int(row["nbytes"] or 0),
    )


def discard_partial_segments(root: str | Path) -> int:
    """Drop unfinished ``*.part.avi`` files left by a previous process."""
    base = Path(root)
    if not base.is_dir():
        return 0
    removed = 0
    for path in base.rglob("*.part.avi"):
        if _unlink_under(base, path):
            removed += 1
    return removed


def adopt_finished_files(store: ArchiveStore, root: str | Path) -> int:
    """Index finished files that never made it into SQLite (crash after write)."""
    base = Path(root)
    if not base.is_dir():
        return 0
    known = store.known_relpaths()
    added = 0
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(".part.avi"):
            continue
        if path.suffix.lower() not in _VIDEO_SUFFIXES:
            continue
        if path.stat().st_size <= 0:
            continue
        rel = path.relative_to(base).as_posix()
        if rel in known:
            continue
        camera_id = path.parent.name
        started = _stamp_from_name(path.name) or datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        )
        ended = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if ended < started:
            ended = started
        store.add_segment(
            camera_id=camera_id,
            relpath=rel,
            started_at=format_utc(started),
            ended_at=format_utc(ended),
            nbytes=path.stat().st_size,
        )
        known.add(rel)
        added += 1
    return added


def prune_archive(
    store: ArchiveStore,
    root: str | Path,
    retention_days: int,
    *,
    now: Optional[datetime] = None,
) -> int:
    """Delete segments older than the retention window. Alert rows are untouched."""
    days = clamp_retention_days(retention_days)
    moment = now or utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    cutoff = moment.astimezone(timezone.utc) - timedelta(days=days)
    cutoff_iso = format_utc(cutoff)
    cutoff_ts = cutoff.timestamp()
    base = Path(root)
    removed = 0
    for segment in store.segments_ended_before(cutoff_iso):
        if _unlink_rel(base, segment.relpath):
            removed += 1
        else:
            removed += 1
        store.delete_segment(segment.id)
    if base.is_dir():
        known = store.known_relpaths()
        for path in list(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(base).as_posix()
            if rel in known:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff_ts:
                continue
            if path.suffix.lower() in _VIDEO_SUFFIXES or path.name.endswith(".part.avi"):
                if _unlink_under(base, path):
                    removed += 1
        _remove_empty_dirs(base)
    return removed


def open_archive_capture(
    camera,
    *,
    project_root: str | Path | None = None,
    allow_webcam: bool = False,
    timeout_sec: float = 8.0,
    rtsp_opener=None,
    webcam_opener=None,
):
    """Open one authorized camera for continuous recording.

    Raises ArchiveSkipped for synthetic MOCK and ArchiveError when the feed
    cannot be opened. Never returns a stand-in frame source.
    """
    source = (getattr(camera, "source_type", "") or "").strip().lower()
    uri = resolve_uri(getattr(camera, "uri", "") or "")
    root = Path(project_root) if project_root else Path.cwd()

    if source == "file":
        if is_mock_uri(uri) or is_mock_uri(getattr(camera, "uri", "")):
            raise ArchiveSkipped("Synthetic MOCK is not archived.")
        path = Path(uri)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise ArchiveError(
                f"Authorized video file not found: {path.name}. "
                "No archive footage was recorded."
            )
        cap = cv2.VideoCapture(str(path))
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            raise ArchiveError(
                f"Could not open {path.name}. No archive footage was recorded."
            )
        return cap

    if source == "rtsp":
        if not uri:
            raise ArchiveError(
                "RTSP URI is missing or the env secret is empty. "
                "No archive footage was recorded."
            )
        opener = rtsp_opener or _open_rtsp
        try:
            return opener(uri, timeout_sec)
        except IngestError as exc:
            text = str(exc).replace("No detections generated.", "").strip()
            raise ArchiveError(f"{text} No archive footage was recorded.") from exc

    if source == "webcam":
        if not allow_webcam:
            raise ArchiveError(
                WEBCAM_REFUSED_MESSAGE.replace(
                    "No detections generated.",
                    "No archive footage was recorded.",
                )
            )
        device = parse_device(uri or getattr(camera, "uri", "") or "0")
        opener = webcam_opener or cv2.VideoCapture
        cap = opener(device)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            raise ArchiveError(
                f"Webcam device {device} is unavailable. "
                "No archive footage was recorded."
            )
        return cap

    raise ArchiveError(
        f"Unknown camera source {source!r}. No archive footage was recorded."
    )


def iter_capture_frames(cap, target_fps: float, stop_check) -> Iterator[tuple[np.ndarray, float]]:
    """Yield frames at about ``target_fps``. Stops when the stream ends."""
    native = 0.0
    try:
        native = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:
        native = 0.0
    if native < 1.0:
        native = max(1.0, float(target_fps))
    interval = max(1, int(round(native / max(0.1, float(target_fps)))))
    playback = max(1.0, native / interval)
    index = 0
    while not stop_check():
        try:
            ok, frame = cap.read()
        except Exception as exc:
            raise ArchiveError(
                f"Stream read failed ({exc}). No archive footage was invented for the gap."
            ) from exc
        if not ok or frame is None:
            break
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            index += 1
            continue
        if index % interval == 0:
            yield frame, playback
        index += 1


class SegmentWriter:
    """Roll short segments with OpenCV. Close finalizes a partial segment."""

    def __init__(
        self,
        root: str | Path,
        camera_id: str,
        store: ArchiveStore,
        *,
        fps: float = DEFAULT_ARCHIVE_FPS,
        segment_sec: float = DEFAULT_SEGMENT_SEC,
        min_free_bytes: int = 0,
        clock=None,
        transcode_timeout: float = 120,
    ) -> None:
        self.root = Path(root)
        self.camera_id = camera_id
        self.token = safe_token(camera_id)
        self.store = store
        self.fps = max(1.0, float(fps))
        self.segment_sec = max(1.0, float(segment_sec))
        self.min_free_bytes = max(0, int(min_free_bytes))
        self.clock = clock or utc_now
        self.transcode_timeout = transcode_timeout
        self._writer = None
        self._part: Optional[Path] = None
        self._started: Optional[datetime] = None
        self._frames = 0
        self._width = 0
        self._height = 0
        self.segments_closed = 0

    def write(self, frame: np.ndarray) -> None:
        now = self._moment()
        if self._writer is not None and self._started is not None:
            elapsed = (now - self._started).total_seconds()
            if elapsed >= self.segment_sec:
                self.close()
        if self._writer is None:
            self._open(frame, now)
        prepared = _fit_frame(frame, self._width, self._height)
        # Some OpenCV builds return None from write(); only False is a failure.
        wrote = self._writer.write(prepared)
        if wrote is False:
            raise ArchiveError(
                "Archive could not write a segment. Recording stopped for this camera. "
                "No footage was invented."
            )
        self._frames += 1

    def close(self) -> Optional[Path]:
        writer = self._writer
        part = self._part
        started = self._started
        frames = self._frames
        self._writer = None
        self._part = None
        self._started = None
        self._frames = 0
        if writer is not None:
            writer.release()
        if part is None or started is None or frames <= 0:
            if part is not None:
                _unlink_under(self.root, part)
            return None
        if not part.is_file() or part.stat().st_size <= 0:
            _unlink_under(self.root, part)
            return None
        final = self._finalize(part)
        if final is None:
            return None
        wall_end = self._moment()
        span = max(frames / self.fps, (wall_end - started).total_seconds())
        ended = started + timedelta(seconds=span)
        rel = final.relative_to(self.root).as_posix()
        self.store.add_segment(
            camera_id=self.camera_id,
            relpath=rel,
            started_at=format_utc(started),
            ended_at=format_utc(ended),
            nbytes=final.stat().st_size,
        )
        self.segments_closed += 1
        logger.info(
            "Archive segment camera=%s path=%s bytes=%s",
            self.camera_id,
            rel,
            final.stat().st_size,
        )
        return final

    def _open(self, frame: np.ndarray, now: datetime) -> None:
        self._require_space()
        height, width = frame.shape[:2]
        self._width = _even(int(width))
        self._height = _even(int(height))
        directory = self.root / self.token
        directory.mkdir(parents=True, exist_ok=True)
        stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        part = directory / f"{stamp}_{uuid.uuid4().hex[:6]}.part.avi"
        writer = cv2.VideoWriter(
            str(part),
            cv2.VideoWriter_fourcc(*"MJPG"),
            float(self.fps),
            (self._width, self._height),
        )
        if not writer.isOpened():
            writer.release()
            _unlink_under(self.root, part)
            raise ArchiveError(
                "Could not open an archive segment file. No footage was invented."
            )
        self._writer = writer
        self._part = part
        self._started = now
        self._frames = 0

    def _finalize(self, part: Path) -> Optional[Path]:
        mp4 = part.with_name(part.name.replace(".part.avi", ".mp4"))
        if transcode_h264(part, mp4, timeout=self.transcode_timeout):
            _unlink_under(self.root, part)
            return mp4
        avi = part.with_name(part.name.replace(".part.avi", ".avi"))
        try:
            os.replace(part, avi)
        except OSError as exc:
            logger.warning("Could not finish archive segment %s: %s", part.name, exc)
            _unlink_under(self.root, part)
            raise ArchiveError(
                "Could not finish an archive segment. No footage was invented."
            ) from exc
        return avi

    def _require_space(self) -> None:
        if self.min_free_bytes <= 0:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            free = shutil.disk_usage(self.root).free
        except OSError as exc:
            raise ArchiveError(
                f"Could not check free disk ({exc}). Archive recording stopped."
            ) from exc
        if free < self.min_free_bytes:
            raise ArchiveDiskFull(
                "Disk is full. Archive recording stopped. No footage was invented."
            )

    def _moment(self) -> datetime:
        moment = self.clock()
        if not isinstance(moment, datetime):
            raise ArchiveError("Archive clock did not return a time.")
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)


def _stamp_from_name(name: str) -> Optional[datetime]:
    match = _STAMP_RE.match(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _even(value: int) -> int:
    size = int(value)
    if size <= 1:
        return 2
    return size if size % 2 == 0 else size - 1


def _fit_frame(image: np.ndarray, width: int, height: int) -> np.ndarray:
    frame = image
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)


def _unlink_rel(root: Path, relpath: str) -> bool:
    if not relpath or ".." in Path(relpath).parts:
        return False
    return _unlink_under(root, root / relpath)


def _unlink_under(root: Path, path: Path) -> bool:
    base = root.resolve()
    try:
        target = path.resolve()
    except OSError:
        return False
    if target != base and base not in target.parents:
        return False
    if not target.is_file():
        return False
    try:
        target.unlink()
    except OSError as exc:
        logger.warning("Could not remove archive file %s: %s", target.name, exc)
        return False
    return True


def _remove_empty_dirs(root: Path) -> None:
    base = root.resolve()
    for directory in sorted(root.rglob("*"), reverse=True):
        if not directory.is_dir():
            continue
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved == base or base not in resolved.parents:
            continue
        try:
            directory.rmdir()
        except OSError:
            continue
