"""Combine sport context, body-aggression, and optional face assists.

Single enrichment point so MOCK, YOLO, and the activity adapter share
the same metadata contract without double-counting cues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from vision.aggression import (
    AggressionAssessment,
    analyze_aggression,
    detections_from_aggression,
)
from vision.detector import Detection
from vision.face_aggression import FaceAggressionResult, analyze_face_aggression
from vision.sports_catalog import infer_sport_context, resolve_sport, sport_display_name


@dataclass
class AssistState:
    prev_bgr: Optional[np.ndarray] = None
    prev_motion: Optional[float] = None


@dataclass
class SceneAssist:
    sport_context: Optional[str] = None
    sport_confidence: float = 0.0
    sport_display: str = ""
    aggression: AggressionAssessment = field(default_factory=AggressionAssessment)
    face: FaceAggressionResult = field(default_factory=FaceAggressionResult)

    def to_extras(self) -> dict:
        extras: dict = {
            "aggression": self.aggression.to_dict(),
            "face_aggression": self.face.to_dict(),
        }
        if self.sport_context:
            extras["sport_context"] = self.sport_context
            extras["sport_confidence"] = round(float(self.sport_confidence), 3)
            extras["sport_display"] = self.sport_display or sport_display_name(
                self.sport_context
            )
        return extras


def _existing_sport(detections: Sequence[Detection]) -> tuple[Optional[str], float]:
    best_id: Optional[str] = None
    best_conf = 0.0
    for det in detections:
        extras = det.extras or {}
        raw = extras.get("sport_context") or extras.get("sport")
        if not raw:
            continue
        entry = resolve_sport(str(raw))
        sport_id = entry.id if entry is not None else str(raw).strip().lower()
        if not sport_id:
            continue
        conf = extras.get("sport_confidence")
        try:
            score = float(conf) if conf is not None else float(det.confidence)
        except (TypeError, ValueError):
            score = float(det.confidence)
        if score >= best_conf:
            best_id, best_conf = sport_id, score
    return best_id, best_conf


def _activity_wants_sport(detections: Sequence[Detection]) -> bool:
    from vision.dataset import canonicalize_category

    for det in detections:
        try:
            if canonicalize_category(det.label) == "game_or_play":
                return True
        except ValueError:
            continue
    return False


def enrich_detections(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
    state: Optional[AssistState] = None,
    *,
    enable_face: Optional[bool] = None,
) -> tuple[List[Detection], SceneAssist]:
    """Attach sport / aggression / face metadata and emit soft fight labels."""
    state = state or AssistState()
    dets = list(detections)
    sport_id, sport_conf = _existing_sport(dets)
    inferred_id, inferred_conf = infer_sport_context(image_bgr)
    if sport_id is None and inferred_id and (
        inferred_conf >= 0.70 or (_activity_wants_sport(dets) and inferred_conf >= 0.50)
    ):
        sport_id, sport_conf = inferred_id, inferred_conf

    aggression = analyze_aggression(
        image_bgr,
        prev_bgr=state.prev_bgr,
        detections=dets,
        prev_motion=state.prev_motion,
    )
    face = analyze_face_aggression(image_bgr, enabled=enable_face)

    assist = SceneAssist(
        sport_context=sport_id,
        sport_confidence=float(sport_conf or 0.0),
        sport_display=sport_display_name(sport_id) if sport_id else "",
        aggression=aggression,
        face=face,
    )
    extras = assist.to_extras()

    merged: List[Detection] = []
    attached = False
    for det in dets:
        payload = dict(det.extras or {})
        payload.update(extras)
        if sport_id and "sport_context" not in (det.extras or {}):
            payload["sport_context"] = sport_id
            payload["sport_confidence"] = assist.sport_confidence
            payload["sport_display"] = assist.sport_display
        merged.append(
            Detection(
                label=det.label,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                extras=payload,
            )
        )
        attached = True
    if not attached:
        merged.append(
            Detection(
                label="scene_assist",
                confidence=max(assist.aggression.score, assist.sport_confidence, 0.2),
                extras=extras,
            )
        )

    existing_labels = {d.label.lower() for d in merged}
    h, w = (image_bgr.shape[:2] + (0, 0))[:2] if image_bgr is not None else (0, 0)
    bbox = (0.0, 0.0, float(w), float(h))
    for extra in detections_from_aggression(aggression, bbox=bbox):
        if extra.label.lower() not in existing_labels:
            extra.extras.update(extras)
            merged.append(extra)
            existing_labels.add(extra.label.lower())

    if image_bgr is not None and getattr(image_bgr, "size", 0) > 0:
        state.prev_bgr = np.array(image_bgr, copy=True)
        state.prev_motion = aggression.motion_intensity
    return merged, assist
