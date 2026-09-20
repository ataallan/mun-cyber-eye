"""Pluggable vision adapters: Phase 3 activity, YOLO, or honest MOCK.

MOCK mode is intentional — it synthesizes detections so the pipeline, risk
engine, and console remain fully demoable without a checkpoint or GPU.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detection from a vision adapter."""

    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    extras: dict[str, Any] = field(default_factory=dict)


class VisionAdapter(ABC):
    """Base interface for vision backends."""

    name: str = "base"

    @abstractmethod
    def detect(self, image_bgr: np.ndarray, frame_index: int = 0) -> List[Detection]:
        """Run detection on a BGR image; return zero or more detections."""


class MockVisionAdapter(VisionAdapter):
    """Deterministic mock detections for demos and tests.

    Cycles through ordinary / fight / fall / weapon-like patterns based on
    frame index so the risk engine and alert UI can be exercised end-to-end.
    """

    name = "mock"

    # Scripted scenarios keyed by (frame_index % period)
    _SCRIPT = {
        0: [],  # ordinary — empty people scene
        1: [Detection("person", 0.91, (40, 50, 180, 400))],
        2: [
            Detection("person", 0.88, (40, 50, 180, 400)),
            Detection("person", 0.86, (200, 60, 340, 410)),
        ],
        3: [
            Detection("person", 0.90, (80, 40, 220, 420)),
            Detection("person", 0.89, (150, 50, 290, 430)),
            Detection("close_proximity", 0.78, (80, 40, 290, 430)),
            Detection("rapid_motion", 0.82, (100, 80, 280, 380)),
        ],
        4: [
            Detection("person", 0.92, (100, 200, 250, 450)),
            Detection("person_down", 0.81, (100, 280, 320, 460)),
            Detection("horizontal_pose", 0.75, (100, 280, 320, 460)),
        ],
        5: [
            Detection("person", 0.93, (60, 40, 200, 420)),
            Detection("knife", 0.71, (180, 180, 230, 260)),
            Detection("raised_object", 0.68, (180, 100, 240, 260)),
        ],
        6: [
            Detection("person", 0.87, (50, 50, 190, 400)),
            Detection("person", 0.85, (210, 55, 350, 405)),
            Detection("rapid_motion", 0.80, (50, 50, 350, 405)),
            Detection("strike_motion", 0.74, (120, 100, 280, 350)),
        ],
        7: [Detection("person", 0.90, (120, 60, 260, 400))],
    }

    def detect(self, image_bgr: np.ndarray, frame_index: int = 0) -> List[Detection]:
        _ = image_bgr  # unused; mock is scripted
        key = frame_index % 8
        dets = list(self._SCRIPT.get(key, []))
        logger.debug("MOCK frame %s → %s detections", frame_index, len(dets))
        return dets


class YoloVisionAdapter(VisionAdapter):
    """Ultralytics YOLO adapter (optional dependency)."""

    name = "yolo"

    # Map COCO / custom labels toward our risk heuristics
    WEAPON_LIKE = {"knife", "scissors", "baseball bat", "sports ball"}  # heuristic
    PERSON_LABELS = {"person"}

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(model_name)
        logger.info("Loaded ultralytics YOLO model: %s", model_name)

    def detect(self, image_bgr: np.ndarray, frame_index: int = 0) -> List[Detection]:
        results = self.model.predict(image_bgr, verbose=False)
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = str(names.get(cls_id, f"class_{cls_id}")).lower()
            xyxy = box.xyxy[0].tolist()
            bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            detections.append(Detection(label=label, confidence=conf, bbox_xyxy=bbox))

            # Enrich with soft signals for risk heuristics
            if label in self.WEAPON_LIKE:
                detections.append(
                    Detection(
                        label="raised_object" if label in {"baseball bat", "knife"} else "suspicious_object",
                        confidence=conf * 0.9,
                        bbox_xyxy=bbox,
                        extras={"source_label": label},
                    )
                )

        # Simple proximity / motion proxy when multiple people
        people = [d for d in detections if d.label == "person"]
        if len(people) >= 2:
            detections.append(
                Detection(
                    label="close_proximity",
                    confidence=0.55,
                    bbox_xyxy=people[0].bbox_xyxy,
                )
            )

        logger.debug("YOLO frame %s → %s detections", frame_index, len(detections))
        return detections


def _try_activity_adapter() -> Optional[VisionAdapter]:
    """Load Phase 3 activity checkpoint when present; never raise to caller."""
    try:
        from vision.activity import try_load_activity_adapter

        return try_load_activity_adapter()
    except Exception as exc:
        logger.info("Activity model not loaded (%s)", exc)
        return None


def create_adapter(backend: Optional[str] = None) -> VisionAdapter:
    """Create a vision adapter.

    backend: auto | activity | yolo | mock (default from VISION_BACKEND env or auto)

    ``auto`` prefers a Phase 3 activity checkpoint, then YOLO, then honest MOCK.
    """
    choice = (backend or os.getenv("VISION_BACKEND", "auto")).strip().lower()

    if choice == "mock":
        logger.info("Vision backend: MOCK (explicit)")
        return MockVisionAdapter()

    if choice == "activity":
        adapter = _try_activity_adapter()
        if adapter is not None:
            logger.info("Vision backend: activity (Phase 3)")
            return adapter
        logger.warning(
            "Activity backend requested but checkpoint unavailable; falling back to MOCK"
        )
        return MockVisionAdapter()

    if choice in {"yolo", "auto"}:
        if choice == "auto":
            adapter = _try_activity_adapter()
            if adapter is not None:
                logger.info("Vision backend: activity (Phase 3 checkpoint found)")
                return adapter
        try:
            adapter = YoloVisionAdapter()
            logger.info("Vision backend: YOLO")
            return adapter
        except Exception as exc:  # ImportError or model load failure
            if choice == "yolo":
                logger.warning(
                    "YOLO requested but unavailable (%s); falling back to MOCK",
                    exc,
                )
            else:
                logger.info(
                    "YOLO not available (%s); using MOCK mode for demo",
                    exc,
                )
            return MockVisionAdapter()

    logger.warning("Unknown VISION_BACKEND=%s; using MOCK", choice)
    return MockVisionAdapter()
