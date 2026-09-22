"""End-to-end pipeline: ingest → vision (Phase 3 activity / YOLO / MOCK) → risk → alerts.

Authorized sources only. Alerts require human review — no autonomous enforcement.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from alerts.notify import NotificationService
from alerts.schema import recommended_human_action, short_rationale
from alerts.store import Alert, AlertStore
from ingest.cameras import Camera, CameraStore
from ingest.clips import (
    DEFAULT_CLIP_MAX_SEC,
    DEFAULT_CLIP_POST_SEC,
    DEFAULT_CLIP_PRE_SEC,
    DEFAULT_MAX_INCIDENT_CLIPS,
    clip_duration_sec,
    prune_incident_clips,
    select_clip_frames,
    write_incident_clip,
)
from ingest.errors import IngestError
from ingest.sampler import FrameSampler, SampledFrame
from ingest.source import iter_camera_frames
from risk.engine import RiskEngine
from vision.object_inventory import collect_object_inventory, merge_inventory_detections
from vision.assists import AssistState, enrich_detections
from vision.detector import Detection, VisionAdapter, create_adapter

logger = logging.getLogger(__name__)


@dataclass
class FrameAssessment:
    """Per-frame risk prediction, including classes that do not create alerts."""

    frame_index: int
    category: str
    confidence: float
    risk_level: str
    should_alert: bool
    rationale: str = ""
    sport_context: str = ""
    sport_display: str = ""
    aggression_score: float = 0.0
    aggression_cues: list[str] = field(default_factory=list)
    face_cue_status: str = "disabled"
    place_type: str = "unknown"
    place_display: str = ""
    place_source: str = "none"
    team_kit_similarity: float = 0.0
    jersey_like_colors: bool = False
    objects_seen: list[dict] = field(default_factory=list)
    objects_backend: str = "unavailable"
    fall_manner: str = ""
    fall_confidence: float = 0.0
    fall_display: str = ""
    gunshot_proxy: bool = False
    gunshot_confidence: float = 0.0
    gunshot_audio_status: str = "disabled"
    aimed_at_person: bool = False
    weapon_use_intensity: float = 0.0
    weapon_use_tier: str = ""
    use_intensity_label: str = ""
    weapon_class: str = ""
    harm_potential: str = ""
    weapon_id: str = ""
    thrown_at_person: bool = False
    throw_label: str = ""
    throw_confidence: float = 0.0


@dataclass
class PipelineResult:
    frames_processed: int
    alerts_created: List[Alert]
    backend: str
    source_label: str
    camera_id: str = ""
    location_label: str = ""
    error: Optional[str] = None
    category_counts: Dict[str, int] = field(default_factory=dict)
    assessments: List[FrameAssessment] = field(default_factory=list)
    objects_backend: str = "unavailable"
    object_counts: Dict[str, int] = field(default_factory=dict)
    objects_note: str = ""


class CyberEyePipeline:
    def __init__(
        self,
        store: AlertStore,
        adapter: Optional[VisionAdapter] = None,
        engine: Optional[RiskEngine] = None,
        snapshot_dir: str | Path = "data/snapshots",
        notifier: Optional[NotificationService] = None,
        location_label: Optional[str] = None,
        camera_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        activity_data_root: Optional[str | Path] = None,
        camera_place_type: Optional[str] = None,
        object_mode: Optional[str] = None,
        notify_extra_recipients: Optional[List[str]] = None,
        clip_dir: Optional[str | Path] = None,
        clip_pre_sec: float = DEFAULT_CLIP_PRE_SEC,
        clip_post_sec: float = DEFAULT_CLIP_POST_SEC,
        clip_max_sec: float = DEFAULT_CLIP_MAX_SEC,
        max_incident_clips: int = DEFAULT_MAX_INCIDENT_CLIPS,
        alert_cooldown_sec: float = 0.0,
    ) -> None:
        self.store = store
        self.adapter = adapter or create_adapter()
        self.engine = engine or RiskEngine()
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.notifier = notifier
        self.location_label = location_label or os.getenv("DEFAULT_LOCATION_LABEL", "")
        self.camera_id = camera_id or os.getenv("DEFAULT_CAMERA_ID", "")
        self.correlation_id = correlation_id
        self.activity_data_root = Path(
            activity_data_root or os.getenv("ACTIVITY_DATA_ROOT", "data/activity")
        )
        self.camera_place_type = camera_place_type or ""
        self.object_mode = (object_mode or os.getenv("OBJECTS_BACKEND", "auto")).strip().lower()
        self.notify_extra_recipients = list(notify_extra_recipients or [])
        self.clip_dir = Path(clip_dir) if clip_dir else None
        if self.clip_dir is not None:
            self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.clip_pre_sec = max(0.0, float(clip_pre_sec))
        self.clip_post_sec = max(0.0, float(clip_post_sec))
        self.clip_max_sec = max(
            self.clip_pre_sec + self.clip_post_sec, float(clip_max_sec)
        )
        self.max_incident_clips = int(max_incident_clips)
        self.alert_cooldown_sec = max(0.0, float(alert_cooldown_sec))
        self._cooldown_until: Dict[tuple[str, str], float] = {}

    def run_video(
        self,
        video_path: str | Path,
        sample_fps: float = 2.0,
        max_frames: Optional[int] = None,
        source_label: Optional[str] = None,
    ) -> PipelineResult:
        label = source_label or os.getenv(
            "DEFAULT_CAMERA_LABEL", "Authorized Camera — Demo Lab"
        )
        sampler = FrameSampler(
            video_path,
            sample_fps=sample_fps,
            max_frames=max_frames,
            source_label=label,
        )
        return self._process(sampler.frames(), source_label=label)

    def run_frames(
        self,
        frames: List[SampledFrame],
        source_label: str = "Authorized Camera — Demo Lab",
        camera_id: Optional[str] = None,
        location_label: Optional[str] = None,
        camera_place_type: Optional[str] = None,
    ) -> PipelineResult:
        return self._process(
            iter(frames),
            source_label=source_label,
            camera_id=camera_id,
            location_label=location_label,
            camera_place_type=camera_place_type,
        )

    def run_camera(
        self,
        camera: Camera,
        *,
        max_frames: Optional[int] = None,
        project_root: str | Path | None = None,
        allow_webcam: Optional[bool] = None,
        timeout_sec: Optional[float] = None,
        rtsp_opener=None,
        webcam_opener=None,
        stop_check: Optional[Callable[[int, bool], bool]] = None,
    ) -> PipelineResult:
        """Run the pipeline on one registered authorized camera."""
        frames = iter_camera_frames(
            camera,
            max_frames=max_frames,
            project_root=project_root,
            allow_webcam=allow_webcam,
            timeout_sec=timeout_sec,
            rtsp_opener=rtsp_opener,
            webcam_opener=webcam_opener,
        )
        return self._process(
            frames,
            source_label=camera.source_label(),
            camera_id=camera.id,
            location_label=camera.location_label,
            camera_place_type=camera.place_type,
            stop_check=stop_check,
        )

    def _process(
        self,
        frame_iter,
        source_label: str,
        camera_id: Optional[str] = None,
        location_label: Optional[str] = None,
        camera_place_type: Optional[str] = None,
        stop_check: Optional[Callable[[int, bool], bool]] = None,
    ) -> PipelineResult:
        alerts: List[Alert] = []
        assessments: List[FrameAssessment] = []
        count = 0
        category_counts: Counter[str] = Counter()
        run_correlation = self.correlation_id or str(uuid.uuid4())
        assist_state = AssistState()
        object_counts: Counter[str] = Counter()
        objects_backend = "unavailable"
        objects_note = ""
        clip_buffer: deque[SampledFrame] = deque()
        open_incident: Optional[dict] = None
        keep_sec = self.clip_pre_sec + self.clip_post_sec
        for frame in frame_iter:
            if stop_check is not None and stop_check(count, open_incident is not None):
                break
            count += 1
            detections = self.adapter.detect(frame.image_bgr, frame_index=frame.index)
            inventory = collect_object_inventory(
                frame.image_bgr,
                detections,
                frame_index=frame.index,
                mode=self.object_mode,
                adapter_name=getattr(self.adapter, "name", "") or "",
            )
            objects_backend = inventory.backend
            objects_note = inventory.note
            detections = merge_inventory_detections(detections, inventory)
            stamp_place = (
                camera_place_type
                if camera_place_type is not None
                else self.camera_place_type
            )
            detections, _assist = enrich_detections(
                frame.image_bgr,
                detections,
                assist_state,
                camera_place_type=stamp_place or None,
                location_label=(
                    location_label if location_label is not None else self.location_label
                ),
                data_root=self.activity_data_root,
            )
            risk = self.engine.assess(detections)
            category_counts[risk.category.value] += 1
            frame_objects = [obj.to_dict() for obj in inventory.objects]
            for obj in inventory.objects:
                object_counts[obj.object_id] += 1
            assessments.append(
                FrameAssessment(
                    frame_index=frame.index,
                    category=risk.category.value,
                    confidence=risk.confidence,
                    risk_level=risk.risk_level.value,
                    should_alert=risk.should_alert,
                    rationale=risk.rationale,
                    sport_context=risk.sport_context or "",
                    sport_display=risk.sport_display or "",
                    aggression_score=risk.aggression_score,
                    aggression_cues=list(risk.aggression_cues),
                    face_cue_status=risk.face_cue_status or "disabled",
                    place_type=risk.place_type or "unknown",
                    place_display=risk.place_display or "",
                    place_source=risk.place_source or "none",
                    team_kit_similarity=risk.team_kit_similarity,
                    jersey_like_colors=bool(risk.jersey_like_colors),
                    objects_seen=frame_objects,
                    objects_backend=inventory.backend,
                    fall_manner=risk.fall_manner or "",
                    fall_confidence=risk.fall_confidence,
                    fall_display=risk.fall_display or "",
                    gunshot_proxy=bool(risk.gunshot_proxy),
                    gunshot_confidence=risk.gunshot_confidence,
                    gunshot_audio_status=risk.gunshot_audio_status or "disabled",
                    aimed_at_person=bool(risk.aimed_at_person),
                    weapon_use_intensity=risk.weapon_use_intensity,
                    weapon_use_tier=risk.weapon_use_tier or "",
                    use_intensity_label=risk.use_intensity_label or "",
                    weapon_class=risk.weapon_class or "",
                    harm_potential=risk.harm_potential or "",
                    weapon_id=risk.weapon_id or "",
                    thrown_at_person=bool(risk.thrown_at_person),
                    throw_label=risk.throw_label or "",
                    throw_confidence=risk.throw_confidence,
                )
            )

            stamp_camera = camera_id if camera_id is not None else self.camera_id
            stamp_location = (
                location_label if location_label is not None else self.location_label
            )
            category_name = (
                risk.category.value
                if hasattr(risk.category, "value")
                else str(risk.category)
            )
            if self.clip_dir is not None:
                clip_buffer.append(_copy_frame(frame))
                cutoff = float(frame.timestamp_sec) - keep_sec
                while clip_buffer and clip_buffer[0].timestamp_sec < cutoff:
                    clip_buffer.popleft()

            if (
                open_incident is not None
                and float(frame.timestamp_sec) + 1e-6 >= float(open_incident["end_ts"])
            ):
                self._close_incident(
                    open_incident,
                    clip_buffer,
                    alerts,
                    source_label=source_label,
                    camera_id=stamp_camera,
                    location_label=stamp_location,
                    correlation_id=run_correlation,
                )
                open_incident = None

            if open_incident is not None:
                if risk.should_alert and category_name == open_incident["category"]:
                    self._extend_incident(open_incident, float(frame.timestamp_sec))
                continue

            if not risk.should_alert:
                continue

            if self._cooling_down(stamp_camera or "", category_name):
                continue

            if self.clip_dir is not None:
                event_ts = float(frame.timestamp_sec)
                open_incident = {
                    "frame": clip_buffer[-1] if clip_buffer else _copy_frame(frame),
                    "detections": detections,
                    "risk": risk,
                    "inventory": inventory,
                    "frame_objects": frame_objects,
                    "event_ts": event_ts,
                    "end_ts": event_ts + self.clip_post_sec,
                    "category": category_name,
                }
                self._extend_incident(open_incident, event_ts)
                continue

            alert = self._emit_alert(
                frame,
                detections,
                risk,
                inventory,
                frame_objects,
                source_label=frame.source_label or source_label,
                camera_id=stamp_camera,
                location_label=stamp_location,
                correlation_id=run_correlation,
            )
            alerts.append(alert)

        if open_incident is not None:
            self._close_incident(
                open_incident,
                clip_buffer,
                alerts,
                source_label=source_label,
                camera_id=camera_id if camera_id is not None else self.camera_id,
                location_label=(
                    location_label if location_label is not None else self.location_label
                ),
                correlation_id=run_correlation,
            )

        return PipelineResult(
            frames_processed=count,
            alerts_created=alerts,
            backend=self.adapter.name,
            source_label=source_label,
            camera_id=(camera_id if camera_id is not None else self.camera_id) or "",
            location_label=(
                location_label if location_label is not None else self.location_label
            )
            or "",
            category_counts=dict(category_counts),
            assessments=assessments,
            objects_backend=objects_backend,
            object_counts=dict(object_counts),
            objects_note=objects_note,
        )

    def _cooling_down(self, camera_id: str, category: str) -> bool:
        if self.alert_cooldown_sec <= 0:
            return False
        until = self._cooldown_until.get((camera_id or "", category))
        if until is None:
            return False
        return time.monotonic() < until

    def _mark_cooldown(self, camera_id: str, category: str) -> None:
        if self.alert_cooldown_sec <= 0:
            return
        self._cooldown_until[(camera_id or "", category)] = (
            time.monotonic() + self.alert_cooldown_sec
        )

    def _extend_incident(self, incident: dict, timestamp_sec: float) -> None:
        max_end = (
            float(incident["event_ts"]) - self.clip_pre_sec + self.clip_max_sec
        )
        proposed = float(timestamp_sec) + self.clip_post_sec
        incident["end_ts"] = min(
            max_end, max(float(incident["end_ts"]), proposed)
        )

    def _close_incident(
        self,
        incident: dict,
        buffer: deque,
        alerts: List[Alert],
        *,
        source_label: str,
        camera_id: Optional[str],
        location_label: Optional[str],
        correlation_id: str,
    ) -> None:
        selected = select_clip_frames(
            buffer,
            event_ts=float(incident["event_ts"]),
            end_ts=float(incident["end_ts"]),
            pre_sec=self.clip_pre_sec,
        )
        if not selected:
            selected = [incident["frame"]]
        clip_path = None
        clip_meta = {
            "pre_sec": self.clip_pre_sec,
            "post_sec": self.clip_post_sec,
            "max_sec": self.clip_max_sec,
            "frames": len(selected),
            "duration_sec": round(clip_duration_sec(selected), 3),
        }
        if self.clip_dir is not None:
            try:
                written = write_incident_clip(
                    selected,
                    self.clip_dir,
                    camera_id=camera_id or "",
                    category=incident["category"],
                )
                if written is not None:
                    clip_path = str(written)
                    prune_incident_clips(self.clip_dir, self.max_incident_clips)
            except Exception as exc:
                logger.warning("Incident clip save failed: %s", exc)
        alert = self._emit_alert(
            incident["frame"],
            incident["detections"],
            incident["risk"],
            incident["inventory"],
            incident["frame_objects"],
            source_label=incident["frame"].source_label or source_label,
            camera_id=camera_id,
            location_label=location_label,
            correlation_id=correlation_id,
            clip_path=clip_path,
            clip_meta=clip_meta,
        )
        self._mark_cooldown(camera_id or "", incident["category"])
        alerts.append(alert)

    def _emit_alert(
        self,
        frame: SampledFrame,
        detections,
        risk,
        inventory,
        frame_objects: list,
        *,
        source_label: str,
        camera_id: Optional[str],
        location_label: Optional[str],
        correlation_id: str,
        clip_path: Optional[str] = None,
        clip_meta: Optional[dict] = None,
    ) -> Alert:
        stamp_camera = camera_id or ""
        stamp_location = location_label or ""
        snap_path = self._save_snapshot(
            frame, risk.category.value, camera_id=stamp_camera
        )
        det_payload = [
            {
                "label": d.label,
                "confidence": d.confidence,
                "bbox_xyxy": list(d.bbox_xyxy),
            }
            for d in detections
        ]
        metadata = {
            "vision_backend": self.adapter.name,
            "contributing_labels": risk.contributing_labels,
            "aggression": {
                "score": risk.aggression_score,
                "cues": list(risk.aggression_cues),
            },
            "face_aggression": {
                "status": risk.face_cue_status,
                "note": risk.face_note,
            },
        }
        if risk.sport_context:
            metadata["sport_context"] = risk.sport_context
            metadata["sport_display"] = risk.sport_display
            metadata["sport_confidence"] = risk.sport_confidence
        if risk.place_type and risk.place_type != "unknown":
            metadata["place_type"] = risk.place_type
            metadata["place_display"] = risk.place_display
            metadata["place_confidence"] = risk.place_confidence
            metadata["place_source"] = risk.place_source
        metadata["kit"] = {
            "team_kit_similarity": risk.team_kit_similarity,
            "jersey_like_colors": bool(risk.jersey_like_colors),
            "note": risk.kit_note
            or (
                "Clothing color clusters are assistive only — not identity "
                "or a guilt label."
            ),
        }
        if stamp_camera:
            metadata["camera_id"] = stamp_camera
        if stamp_location:
            metadata["location_label"] = stamp_location
        ckpt = getattr(self.adapter, "checkpoint_path", None)
        if ckpt is not None:
            metadata["activity_checkpoint"] = str(ckpt)
        scores = next(
            (
                d.extras.get("scores")
                for d in detections
                if isinstance(d.extras, dict) and d.extras.get("scores")
            ),
            None,
        )
        if scores:
            metadata["activity_scores"] = scores
        metadata["objects_backend"] = inventory.backend
        if frame_objects:
            metadata["objects"] = frame_objects
        elif inventory.backend == "unavailable":
            metadata["objects"] = []
            metadata["objects_note"] = inventory.note
        if risk.fall_manner or risk.category == "potential_fall":
            metadata["fall_manner"] = risk.fall_manner or "unknown_fall"
            metadata["fall_confidence"] = risk.fall_confidence
            metadata["fall_display"] = risk.fall_display
        metadata["gunshot"] = {
            "video_proxy": bool(risk.gunshot_proxy),
            "confidence": risk.gunshot_confidence,
            "audio_status": risk.gunshot_audio_status or "disabled",
        }
        if risk.aimed_at_person or risk.weapon_use_tier or risk.weapon_id or risk.weapon_class:
            metadata["weapon"] = {
                "aimed_at_person": bool(risk.aimed_at_person),
                "use_intensity": risk.weapon_use_intensity,
                "use_intensity_label": risk.use_intensity_label or risk.weapon_use_tier,
                "use_tier": risk.weapon_use_tier,
                "weapon_class": risk.weapon_class,
                "harm_potential": risk.harm_potential,
                "weapon_id": risk.weapon_id,
                "cue": (
                    "firearm_aimed_at_person"
                    if risk.aimed_at_person
                    else risk.use_intensity_label or risk.weapon_use_tier
                ),
            }
        if risk.thrown_at_person:
            metadata["throw"] = {
                "thrown_at_person": True,
                "object_label": risk.throw_label,
                "use_tier": "thrown_projectile",
                "confidence": risk.throw_confidence,
                "cue": "object_thrown_at_person",
            }
        if clip_path or clip_meta:
            metadata["incident_clip"] = dict(clip_meta or {})
            if clip_path:
                metadata["incident_clip"]["path"] = clip_path

        alert = self.store.create_alert(
            source_label=source_label,
            category=risk.category.value,
            risk_level=risk.risk_level.value,
            confidence=risk.confidence,
            rationale=risk.rationale,
            frame_index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            snapshot_path=str(snap_path) if snap_path else None,
            detections=det_payload,
            metadata=metadata,
            location_label=stamp_location,
            camera_id=stamp_camera,
            short_rationale_text=short_rationale(risk.rationale),
            recommended_action=recommended_human_action(risk.category.value),
            correlation_id=correlation_id,
            clip_path=clip_path,
        )
        if self.notifier:
            alert = self.notifier.deliver(
                alert,
                actor="system",
                extra_recipients=self.notify_extra_recipients,
            )
        logger.info(
            "Alert %s [%s/%s] conf=%.2f frame=%s clip=%s",
            alert.id[:8],
            risk.risk_level.value,
            risk.category.value,
            risk.confidence,
            frame.index,
            "yes" if clip_path else "no",
        )
        return alert

    def _save_snapshot(
        self,
        frame: SampledFrame,
        category: str,
        camera_id: str = "",
    ) -> Optional[Path]:
        try:
            cam = "".join(
                ch if ch.isalnum() or ch in "-_" else "-"
                for ch in (camera_id or "cam")
            )[:24] or "cam"
            name = f"{cam}_frame_{frame.index:05d}_{category}.jpg"
            path = self.snapshot_dir / name
            ok = cv2.imwrite(str(path), frame.image_bgr)
            return path if ok else None
        except Exception as exc:
            logger.warning("Snapshot save failed: %s", exc)
            return None


def _copy_frame(frame: SampledFrame) -> SampledFrame:
    image = frame.image_bgr
    try:
        image = image.copy()
    except Exception:
        pass
    return SampledFrame(
        index=frame.index,
        timestamp_sec=frame.timestamp_sec,
        image_bgr=image,
        source_label=frame.source_label,
    )


def demo_synthetic_run(
    store: AlertStore,
    frames: int = 16,
    notifier: Optional[NotificationService] = None,
    snapshot_dir: str | Path = "data/snapshots",
    notify_extra_recipients: Optional[List[str]] = None,
) -> PipelineResult:
    """Run MOCK detections over blank frames — no video file required."""
    import numpy as np

    from vision.detector import MockVisionAdapter

    adapter = MockVisionAdapter()
    pipeline = CyberEyePipeline(
        store=store,
        adapter=adapter,
        notifier=notifier,
        snapshot_dir=snapshot_dir,
        object_mode="mock",
        notify_extra_recipients=notify_extra_recipients,
    )
    synthetic = []
    for i in range(frames):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Subtle pattern so JPEGs aren't identical empty blacks
        img[:, :, 0] = (i * 17) % 255
        img[:, :, 1] = 40
        img[:, :, 2] = 60
        synthetic.append(
            SampledFrame(
                index=i,
                timestamp_sec=i / 2.0,
                image_bgr=img,
                source_label="Synthetic sample",
            )
        )
    return pipeline.run_frames(
        synthetic, source_label="Synthetic sample"
    )


def demo_activity_run(
    store: AlertStore,
    checkpoint: Optional[str | Path] = None,
    per_class: int = 2,
    notifier: Optional[NotificationService] = None,
    snapshot_dir: str | Path = "data/snapshots",
    notify_extra_recipients: Optional[List[str]] = None,
) -> PipelineResult:
    """Run the Phase 3 activity adapter on synthetic class-typical frames.

    Falls back to MOCK if the checkpoint cannot be loaded — human review
    remains required either way.
    """
    from vision.activity import try_load_activity_adapter
    from vision.dataset import ACTIVITY_CATEGORIES, render_demo_frame
    from vision.detector import MockVisionAdapter
    from vision.scene_context import DEMO_PLACE_SCENES
    from vision.sports_catalog import DEMO_SPORT_SCENES

    adapter = try_load_activity_adapter(checkpoint)
    if adapter is None:
        logger.warning(
            "Activity demo: checkpoint unavailable; falling back to MOCK heuristics"
        )
        adapter = MockVisionAdapter()

    pipeline = CyberEyePipeline(
        store=store,
        adapter=adapter,
        notifier=notifier,
        snapshot_dir=snapshot_dir,
        notify_extra_recipients=notify_extra_recipients,
    )
    synthetic = []
    idx = 0
    for cat in ACTIVITY_CATEGORIES:
        for k in range(per_class):
            img = render_demo_frame(cat, seed=100 + idx * 3)
            synthetic.append(
                SampledFrame(
                    index=idx,
                    timestamp_sec=idx / 2.0,
                    image_bgr=img,
                    source_label="Activity sample",
                )
            )
            idx += 1
    for sport in DEMO_SPORT_SCENES:
        img = render_demo_frame("game_or_play", seed=400 + idx * 3, sport_context=sport)
        synthetic.append(
            SampledFrame(
                index=idx,
                timestamp_sec=idx / 2.0,
                image_bgr=img,
                source_label="Activity sample",
            )
        )
        idx += 1
    for place in DEMO_PLACE_SCENES:
        img = render_demo_frame(
            "ordinary" if place not in {
                "basketball_court",
                "sports_field",
                "tennis_court",
                "volleyball_court",
                "playground",
                "track",
                "swimming_pool",
                "skate_park",
                "gymnasium",
                "indoor_arena",
            } else "game_or_play",
            seed=700 + idx * 3,
            place_type=place,
        )
        synthetic.append(
            SampledFrame(
                index=idx,
                timestamp_sec=idx / 2.0,
                image_bgr=img,
                source_label="Activity sample",
            )
        )
        idx += 1
    return pipeline.run_frames(
        synthetic, source_label="Activity sample"
    )


def run_registered_cameras(
    pipeline: CyberEyePipeline,
    cameras: List[Camera],
    camera_store: CameraStore,
    *,
    max_frames: Optional[int] = None,
    project_root: str | Path | None = None,
    allow_webcam: Optional[bool] = None,
    timeout_sec: Optional[float] = None,
    rtsp_opener=None,
    webcam_opener=None,
    stop_check: Optional[Callable[[int, bool], bool]] = None,
) -> List[PipelineResult]:
    """Run enabled cameras sequentially (prototype: not true parallel streaming).

    Each camera failure is recorded on that row. The batch continues so one
    offline feed cannot crash the console or invent detections.
    """
    results: List[PipelineResult] = []
    for camera in cameras:
        try:
            result = pipeline.run_camera(
                camera,
                max_frames=max_frames,
                project_root=project_root,
                allow_webcam=allow_webcam,
                timeout_sec=timeout_sec,
                rtsp_opener=rtsp_opener,
                webcam_opener=webcam_opener,
                stop_check=stop_check,
            )
            if result.frames_processed <= 0:
                raise IngestError(
                    "Camera produced no frames. No detections generated."
                )
            camera_store.record_success(camera.id)
            results.append(result)
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            logger.warning("Camera %s (%s) failed: %s", camera.id, camera.name, message)
            try:
                camera_store.record_error(camera.id, message)
            except KeyError:
                pass
            results.append(
                PipelineResult(
                    frames_processed=0,
                    alerts_created=[],
                    backend=pipeline.adapter.name,
                    source_label=camera.source_label(),
                    camera_id=camera.id,
                    location_label=camera.location_label,
                    error=message,
                    objects_backend="unavailable",
                    objects_note="No frames from this camera.",
                )
            )
    return results
