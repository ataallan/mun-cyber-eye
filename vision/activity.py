"""Trainable Phase 3 activity-recognition adapter (OpenCV features + sklearn).

Loads a joblib checkpoint when present. If the checkpoint or sklearn is
missing, callers should fall back to Phase 2 YOLO heuristics or MOCK.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from vision.dataset import ACTIVITY_CATEGORIES
from vision.detector import Detection, VisionAdapter
from vision.features import FEATURE_DIM, FEATURE_VERSION, extract_frame_features

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_REL = Path("data/checkpoints/activity_demo.joblib")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_checkpoint_path() -> Path:
    env = os.getenv("ACTIVITY_CHECKPOINT", "").strip()
    if env:
        return Path(env)
    return project_root() / DEFAULT_CHECKPOINT_REL


@dataclass
class ActivityCheckpoint:
    """Serializable bundle stored at the demo/default checkpoint path."""

    model: Any
    scaler: Any
    categories: List[str] = field(default_factory=lambda: list(ACTIVITY_CATEGORIES))
    feature_version: int = FEATURE_VERSION
    feature_dim: int = FEATURE_DIM
    metrics: dict = field(default_factory=dict)
    created_at: str = ""
    notes: str = (
        "Demo activity model (OpenCV features + sklearn). "
        "Trained for CI/demo — not production human-activity recognition."
    )
    sklearn_version: str = ""
    model_type: str = "forest"

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "scaler": self.scaler,
            "categories": list(self.categories),
            "feature_version": self.feature_version,
            "feature_dim": self.feature_dim,
            "metrics": dict(self.metrics),
            "created_at": self.created_at,
            "notes": self.notes,
            "sklearn_version": self.sklearn_version,
            "model_type": self.model_type,
            "backend": "opencv_sklearn",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ActivityCheckpoint":
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("Not an activity checkpoint (missing model)")
        return cls(
            model=payload["model"],
            scaler=payload.get("scaler"),
            categories=list(payload.get("categories") or ACTIVITY_CATEGORIES),
            feature_version=int(payload.get("feature_version", -1)),
            feature_dim=int(payload.get("feature_dim", -1)),
            metrics=dict(payload.get("metrics") or {}),
            created_at=str(payload.get("created_at") or ""),
            notes=str(payload.get("notes") or ""),
            sklearn_version=str(payload.get("sklearn_version") or ""),
            model_type=str(payload.get("model_type") or "forest"),
        )


def save_checkpoint(bundle: ActivityCheckpoint, path: str | Path) -> Path:
    joblib = _require_joblib()
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not bundle.created_at:
        bundle.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    joblib.dump(bundle.to_dict(), dest, protocol=4)
    logger.info("Wrote activity checkpoint %s", dest)
    return dest


def load_checkpoint(path: str | Path) -> ActivityCheckpoint:
    joblib = _require_joblib()
    payload = joblib.load(path)
    bundle = ActivityCheckpoint.from_dict(payload)
    if bundle.feature_version != FEATURE_VERSION:
        raise ValueError(
            f"Checkpoint feature_version {bundle.feature_version} != {FEATURE_VERSION}"
        )
    if bundle.feature_dim != FEATURE_DIM:
        raise ValueError(
            f"Checkpoint feature_dim {bundle.feature_dim} != {FEATURE_DIM}"
        )
    return bundle


def _require_joblib():
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "joblib/scikit-learn is required to load the Phase 3 activity model"
        ) from exc
    return joblib


class ActivityVisionAdapter(VisionAdapter):
    """Classify a frame into proposal activity categories.

    Emits a detection whose label is the predicted category plus a soft
    ``person`` box so the console still shows a scene actor. The risk engine
    treats canonical category labels as first-class Phase 3 decisions.
    """

    name = "activity"

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        bundle = load_checkpoint(self.checkpoint_path)
        self.model = bundle.model
        self.scaler = bundle.scaler
        self.categories = list(bundle.categories)
        self.bundle = bundle
        self._prev: Optional[np.ndarray] = None
        logger.info(
            "Loaded Phase 3 activity checkpoint %s (model=%s)",
            self.checkpoint_path,
            bundle.model_type,
        )

    def detect(self, image_bgr: np.ndarray, frame_index: int = 0) -> List[Detection]:
        feat = extract_frame_features(image_bgr, self._prev)
        self._prev = np.array(image_bgr, copy=True)
        x = feat.reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        classes = [str(c) for c in getattr(self.model, "classes_", self.categories)]
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            idx = int(np.argmax(proba))
            scores = {classes[i]: float(proba[i]) for i in range(len(classes))}
            confidence = float(proba[idx])
            label = classes[idx]
        else:
            label = str(self.model.predict(x)[0])
            confidence = 0.6
            scores = {label: confidence}

        extras = {
            "source": "phase3_activity",
            "scores": scores,
            "feature_version": FEATURE_VERSION,
            "checkpoint": str(self.checkpoint_path),
            "frame_index": frame_index,
        }
        h, w = image_bgr.shape[:2]
        person_box = (w * 0.25, h * 0.15, w * 0.75, h * 0.95)
        return [
            Detection(
                label=label,
                confidence=confidence,
                bbox_xyxy=person_box,
                extras=extras,
            ),
            Detection(
                label="person",
                confidence=min(0.92, max(0.4, confidence)),
                bbox_xyxy=person_box,
                extras={"source": "phase3_activity_context"},
            ),
        ]


def try_load_activity_adapter(
    path: Optional[str | Path] = None,
) -> Optional[ActivityVisionAdapter]:
    """Return an adapter if a valid checkpoint can be loaded, else None."""
    ckpt = Path(path) if path is not None else default_checkpoint_path()
    if not ckpt.is_file():
        logger.info("No activity checkpoint at %s", ckpt)
        return None
    try:
        return ActivityVisionAdapter(ckpt)
    except Exception as exc:  # missing sklearn, corrupt file, version mismatch
        logger.warning("Failed to load activity checkpoint %s: %s", ckpt, exc)
        return None


def predict_category(
    adapter: ActivityVisionAdapter,
    image_bgr: np.ndarray,
) -> tuple[str, float, dict]:
    dets = adapter.detect(image_bgr, frame_index=0)
    top = next(d for d in dets if d.label in ACTIVITY_CATEGORIES or d.label != "person")
    return top.label, top.confidence, dict(top.extras or {})
