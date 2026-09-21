"""CPU OpenCV proxies for body signs of aggressiveness.

These cues are **assistive**. They do not determine assault, intent, or
identity. High motion during sport is often intense play, not a fight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

from vision.detector import Detection

# Labels the risk engine already treats as fight-adjacent.
AGGRESSION_LABELS = (
    "aggressive_motion",
    "aggressive_pose",
    "rapid_motion",
    "strike_motion",
    "close_proximity",
)


@dataclass
class AggressionAssessment:
    """Soft body-aggression proxies for one frame pair."""

    score: float = 0.0
    cues: List[str] = field(default_factory=list)
    motion_intensity: float = 0.0
    sudden_acceleration: float = 0.0
    opposing_motion: float = 0.0
    raised_arm: float = 0.0
    proximity: float = 0.0
    uniform_scene_change: bool = False
    notes: str = (
        "OpenCV motion/pose proxies only — not a determination of assault."
    )

    def to_dict(self) -> dict:
        return {
            "score": round(float(self.score), 3),
            "cues": list(self.cues),
            "motion_intensity": round(float(self.motion_intensity), 3),
            "sudden_acceleration": round(float(self.sudden_acceleration), 3),
            "opposing_motion": round(float(self.opposing_motion), 3),
            "raised_arm": round(float(self.raised_arm), 3),
            "proximity": round(float(self.proximity), 3),
            "uniform_scene_change": bool(self.uniform_scene_change),
            "notes": self.notes,
        }


def _as_gray(image: np.ndarray, size: tuple[int, int] = (160, 120)) -> np.ndarray:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
    small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _person_proximity(detections: Sequence[Detection] | None) -> float:
    people = [
        d
        for d in (detections or [])
        if str(getattr(d, "label", "")).lower() == "person"
    ]
    if len(people) < 2:
        return 0.0
    centers = []
    widths = []
    for p in people:
        x1, y1, x2, y2 = p.bbox_xyxy
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        widths.append(max(8.0, abs(x2 - x1)))
    best = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = float(np.hypot(dx, dy))
            scale = 0.5 * (widths[i] + widths[j])
            overlap = max(0.0, 1.0 - dist / max(scale * 2.2, 1.0))
            best = max(best, overlap)
    return float(min(1.0, best))


def analyze_aggression(
    image_bgr: np.ndarray,
    prev_bgr: Optional[np.ndarray] = None,
    detections: Sequence[Detection] | None = None,
    prev_motion: Optional[float] = None,
) -> AggressionAssessment:
    """Compare this frame to the previous one and emit soft aggression cues."""
    result = AggressionAssessment()
    result.proximity = _person_proximity(detections)
    if (
        prev_bgr is None
        or getattr(prev_bgr, "size", 0) == 0
        or image_bgr is None
        or getattr(image_bgr, "size", 0) == 0
    ):
        if result.proximity >= 0.55:
            result.cues.append("close_proximity")
            result.score = min(0.45, 0.25 + 0.3 * result.proximity)
        return result

    gray = _as_gray(image_bgr)
    prev = _as_gray(prev_bgr)
    if gray.shape != prev.shape:
        prev = cv2.resize(prev, (gray.shape[1], gray.shape[0]))
    diff = cv2.absdiff(gray, prev)
    motion_mean = float(diff.mean()) / 255.0
    motion_frac = float((diff > 18).mean())
    result.motion_intensity = float(min(1.0, 0.55 * motion_mean + 0.45 * motion_frac))

    # 4×4 grid: a near-uniform change is a scene cut / lighting shift, not a clash.
    h, w = diff.shape
    cells = []
    for y in range(4):
        for x in range(4):
            block = diff[y * h // 4 : (y + 1) * h // 4, x * w // 4 : (x + 1) * w // 4]
            cells.append(float(block.mean()))
    cell_std = float(np.std(cells)) if cells else 0.0
    cell_mean = float(np.mean(cells)) if cells else 0.0
    result.uniform_scene_change = bool(cell_mean > 18.0 and cell_std < 10.0)
    if result.uniform_scene_change:
        result.motion_intensity *= 0.28
        result.notes = (
            "Mostly uniform frame change (scene cut or lighting) — "
            "not treated as body aggression."
        )

    if prev_motion is not None:
        result.sudden_acceleration = float(
            min(1.0, abs(result.motion_intensity - float(prev_motion)) * 1.4)
        )

    mid = w // 2
    left = float(diff[:, :mid].mean())
    right = float(diff[:, mid:].mean())
    if left > 10 and right > 10 and not result.uniform_scene_change:
        # Localized motion on both sides is a clash proxy; uniform already gated.
        result.opposing_motion = float(min(1.0, ((left + right) / 2.0) / 55.0))
        # Prefer approaching centroids (people moving toward each other).
        left_col = np.where(diff[:, :mid] > 18)
        right_col = np.where(diff[:, mid:] > 18)
        if left_col[1].size and right_col[1].size:
            lc = float(left_col[1].mean())
            rc = float(right_col[1].mean()) + mid
            gap = abs(rc - lc) / float(max(w, 1))
            result.opposing_motion = float(min(1.0, result.opposing_motion + (1.0 - gap) * 0.35))

    upper = diff[: max(1, h // 3), :]
    result.raised_arm = float((upper > 22).mean())
    if result.uniform_scene_change:
        result.opposing_motion *= 0.2
        result.raised_arm *= 0.25
        result.sudden_acceleration *= 0.35

    result.score = float(
        min(
            1.0,
            0.32 * result.motion_intensity
            + 0.18 * result.sudden_acceleration
            + 0.22 * result.opposing_motion
            + 0.16 * result.raised_arm
            + 0.12 * result.proximity,
        )
    )

    cues: list[str] = []
    if result.motion_intensity >= 0.12 and not result.uniform_scene_change:
        cues.append("aggressive_motion")
    if result.motion_intensity >= 0.35 and not result.uniform_scene_change:
        cues.append("rapid_motion")
    if result.raised_arm >= 0.12 and not result.uniform_scene_change:
        cues.append("aggressive_pose")
        cues.append("strike_motion")
    if result.proximity >= 0.40 and (
        result.opposing_motion >= 0.18 or result.motion_intensity >= 0.12
    ):
        cues.append("close_proximity")
    if result.score >= 0.20 and not cues and not result.uniform_scene_change:
        cues.append("aggressive_motion")
    if result.score < 0.16:
        cues = []
    result.cues = cues
    return result


def detections_from_aggression(
    assessment: AggressionAssessment,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> List[Detection]:
    """Emit soft labels the risk engine already knows (FIGHT_SIGNALS)."""
    if assessment.score < 0.18 or not assessment.cues:
        return []
    extras = {"source": "aggression_proxy", "aggression": assessment.to_dict()}
    out: List[Detection] = []
    conf_map = {
        "aggressive_motion": max(0.4, assessment.motion_intensity),
        "rapid_motion": max(0.4, assessment.motion_intensity),
        "aggressive_pose": max(0.4, assessment.raised_arm),
        "strike_motion": max(0.4, assessment.raised_arm),
        "close_proximity": max(0.4, assessment.proximity),
    }
    for label in assessment.cues:
        out.append(
            Detection(
                label=label,
                confidence=float(min(0.9, conf_map.get(label, assessment.score))),
                bbox_xyxy=bbox,
                extras=dict(extras),
            )
        )
    return out
