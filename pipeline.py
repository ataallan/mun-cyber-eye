"""End-to-end pipeline: ingest → vision (Phase 3 activity / YOLO / MOCK) → risk → alerts.

Authorized sources only. Alerts require human review — no autonomous enforcement.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2

from alerts.store import Alert, AlertStore
from ingest.sampler import FrameSampler, SampledFrame
from risk.engine import RiskEngine
from vision.detector import Detection, VisionAdapter, create_adapter

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    frames_processed: int
    alerts_created: List[Alert]
    backend: str
    source_label: str


class CyberEyePipeline:
    def __init__(
        self,
        store: AlertStore,
        adapter: Optional[VisionAdapter] = None,
        engine: Optional[RiskEngine] = None,
        snapshot_dir: str | Path = "data/snapshots",
    ) -> None:
        self.store = store
        self.adapter = adapter or create_adapter()
        self.engine = engine or RiskEngine()
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

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
    ) -> PipelineResult:
        return self._process(iter(frames), source_label=source_label)

    def _process(self, frame_iter, source_label: str) -> PipelineResult:
        alerts: List[Alert] = []
        count = 0
        for frame in frame_iter:
            count += 1
            detections = self.adapter.detect(frame.image_bgr, frame_index=frame.index)
            risk = self.engine.assess(detections)

            if not risk.should_alert:
                continue

            snap_path = self._save_snapshot(frame, risk.category.value)
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
            }
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

            alert = self.store.create_alert(
                source_label=frame.source_label or source_label,
                category=risk.category.value,
                risk_level=risk.risk_level.value,
                confidence=risk.confidence,
                rationale=risk.rationale,
                frame_index=frame.index,
                timestamp_sec=frame.timestamp_sec,
                snapshot_path=str(snap_path) if snap_path else None,
                detections=det_payload,
                metadata=metadata,
            )
            alerts.append(alert)
            logger.info(
                "Alert %s [%s/%s] conf=%.2f frame=%s",
                alert.id[:8],
                risk.risk_level.value,
                risk.category.value,
                risk.confidence,
                frame.index,
            )

        return PipelineResult(
            frames_processed=count,
            alerts_created=alerts,
            backend=self.adapter.name,
            source_label=source_label,
        )

    def _save_snapshot(self, frame: SampledFrame, category: str) -> Optional[Path]:
        try:
            name = f"frame_{frame.index:05d}_{category}.jpg"
            path = self.snapshot_dir / name
            ok = cv2.imwrite(str(path), frame.image_bgr)
            return path if ok else None
        except Exception as exc:
            logger.warning("Snapshot save failed: %s", exc)
            return None


def demo_synthetic_run(store: AlertStore, frames: int = 16) -> PipelineResult:
    """Run MOCK detections over blank frames — no video file required."""
    import numpy as np

    from vision.detector import MockVisionAdapter

    adapter = MockVisionAdapter()
    pipeline = CyberEyePipeline(store=store, adapter=adapter)
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
                source_label="Authorized Camera — Synthetic Demo",
            )
        )
    return pipeline.run_frames(synthetic)


def demo_activity_run(
    store: AlertStore,
    checkpoint: Optional[str | Path] = None,
    per_class: int = 2,
) -> PipelineResult:
    """Run the Phase 3 activity adapter on synthetic class-typical frames.

    Falls back to MOCK if the checkpoint cannot be loaded — human review
    remains required either way.
    """
    from vision.activity import try_load_activity_adapter
    from vision.dataset import ACTIVITY_CATEGORIES, render_demo_frame
    from vision.detector import MockVisionAdapter

    adapter = try_load_activity_adapter(checkpoint)
    if adapter is None:
        logger.warning(
            "Activity demo: checkpoint unavailable; falling back to MOCK heuristics"
        )
        adapter = MockVisionAdapter()

    pipeline = CyberEyePipeline(store=store, adapter=adapter)
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
                    source_label="Authorized Camera — Activity Demo",
                )
            )
            idx += 1
    return pipeline.run_frames(synthetic)
