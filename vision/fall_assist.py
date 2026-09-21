"""Fall *manner* assist on top of ``potential_fall``.

The alert category stays ``potential_fall`` (person-down still needs review).
This module only labels a **subtype**: sudden/unprecedented collapse vs an
accidental-looking trip/slip vs unknown. It cannot medically diagnose
syncope, assault, or a trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from vision.aggression import AggressionAssessment
from vision.detector import Detection

FALL_LABELS = frozenset(
    {
        "person_down",
        "horizontal_pose",
        "fall",
        "potential_fall",
        "collapse",
    }
)
WEAPON_CONTEXT_LABELS = frozenset(
    {
        "knife",
        "gun",
        "firearm",
        "pistol",
        "rifle",
        "handgun",
        "raised_object",
        "possible_gunshot_video_proxy",
        "firearm_aimed_at_person",
        "weapon_pointed_at_person",
    }
)

SUDDEN_COLLAPSE = "sudden_collapse"
ACCIDENTAL_FALL = "accidental_fall"
UNKNOWN_FALL = "unknown_fall"
UNPRECEDENTED_FALL = "unprecedented_fall"  # alias of sudden_collapse

FALL_MANNER_DISPLAY = {
    SUDDEN_COLLAPSE: "unprecedented / sudden collapse",
    UNPRECEDENTED_FALL: "unprecedented / sudden collapse",
    ACCIDENTAL_FALL: "accidental-looking fall (trip/slip-like)",
    UNKNOWN_FALL: "fall manner unknown",
}

HONEST_NOTE = (
    "Fall manner is an OpenCV / bbox proxy only — not a medical diagnosis "
    "of syncope, assault, or a trip. Humans verify every person-down alert."
)


@dataclass
class FallAssessment:
    """Assistive subtype for a potential fall / person-down frame."""

    manner: str = ""
    confidence: float = 0.0
    display: str = ""
    cues: list[str] = field(default_factory=list)
    note: str = HONEST_NOTE
    applicable: bool = False

    def to_dict(self) -> dict:
        return {
            "fall_manner": self.manner or "",
            "fall_confidence": round(float(self.confidence), 3),
            "fall_display": self.display
            or FALL_MANNER_DISPLAY.get(self.manner, self.manner),
            "cues": list(self.cues),
            "note": self.note,
            "applicable": bool(self.applicable),
        }


def _box_wh(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(1.0, abs(float(x2) - float(x1))), max(1.0, abs(float(y2) - float(y1)))


def _aspect_hw(bbox: tuple[float, float, float, float]) -> float:
    w, h = _box_wh(bbox)
    return h / w


def _center_y(bbox: tuple[float, float, float, float]) -> float:
    return (float(bbox[1]) + float(bbox[3])) / 2.0


def _is_standing_box(bbox: tuple[float, float, float, float]) -> bool:
    return _aspect_hw(bbox) >= 1.35


def _is_horizontal_box(bbox: tuple[float, float, float, float]) -> bool:
    return _aspect_hw(bbox) <= 0.85


def fall_labels_present(detections: Sequence[Detection]) -> bool:
    for det in detections:
        if str(det.label).lower() in FALL_LABELS:
            return True
    return False


def analyze_fall_manner(
    detections: Sequence[Detection],
    aggression: Optional[AggressionAssessment] = None,
    prev_person_boxes: Optional[Sequence[tuple[float, float, float, float]]] = None,
) -> FallAssessment:
    """Subtype a person-down / fall frame. Empty manner when no fall cues."""
    labels = {str(d.label).lower() for d in detections}
    fall_hits = labels & FALL_LABELS
    if not fall_hits:
        return FallAssessment()

    aggression = aggression or AggressionAssessment()
    people = [d for d in detections if str(d.label).lower() == "person"]
    down = [d for d in detections if str(d.label).lower() in FALL_LABELS]
    weapon_context = bool(labels & WEAPON_CONTEXT_LABELS)
    cues: list[str] = sorted(fall_hits)

    prev_standing = False
    vertical_drop = False
    if prev_person_boxes:
        prev_standing = any(_is_standing_box(b) for b in prev_person_boxes)
        if down:
            prev_cy = min(_center_y(b) for b in prev_person_boxes)
            now_cy = min(_center_y(d.bbox_xyxy) for d in down)
            # Image Y grows downward; a collapse moves the body lower.
            vertical_drop = now_cy > prev_cy + 12
            if vertical_drop:
                cues.append("vertical_drop")
        if prev_standing:
            cues.append("was_standing")

    horizontal_now = any(
        _is_horizontal_box(d.bbox_xyxy) for d in list(down) + people
    ) or "horizontal_pose" in labels
    if horizontal_now:
        cues.append("horizontal_pose_now")

    sudden = float(aggression.sudden_acceleration) >= 0.45 or float(
        aggression.motion_intensity
    ) >= 0.40
    if sudden:
        cues.append("motion_discontinuity")

    high_aggr = float(aggression.score) >= 0.65

    # Sudden / unprecedented: standing then down, abrupt motion, vertical drop.
    sudden_score = 0.0
    if prev_standing and fall_hits:
        sudden_score += 0.34
    if vertical_drop:
        sudden_score += 0.22
    if sudden:
        sudden_score += 0.28
    if high_aggr:
        sudden_score += 0.10
    if "person_down" in labels and prev_standing:
        sudden_score += 0.08

    # Accidental / trip-like: already wide pose, milder motion, no weapon.
    accidental_score = 0.0
    if horizontal_now:
        accidental_score += 0.32
    if "horizontal_pose" in labels:
        accidental_score += 0.18
    if not high_aggr:
        accidental_score += 0.22
    if not sudden:
        accidental_score += 0.16
    if not weapon_context:
        accidental_score += 0.14
    if prev_standing and sudden:
        accidental_score *= 0.45  # standing-then-drop is not a slow trip

    sudden_score = float(min(0.92, sudden_score))
    accidental_score = float(min(0.88, accidental_score))

    if sudden_score >= 0.55 and sudden_score >= accidental_score + 0.08:
        manner = SUDDEN_COLLAPSE
        conf = sudden_score
    elif accidental_score >= 0.50 and accidental_score >= sudden_score + 0.05:
        manner = ACCIDENTAL_FALL
        conf = accidental_score
    else:
        manner = UNKNOWN_FALL
        conf = max(0.35, min(sudden_score, accidental_score, 0.48))

    return FallAssessment(
        manner=manner,
        confidence=round(conf, 3),
        display=FALL_MANNER_DISPLAY[manner],
        cues=cues,
        applicable=True,
    )


def person_boxes_from_detections(
    detections: Sequence[Detection],
) -> list[tuple[float, float, float, float]]:
    boxes = []
    for det in detections:
        if str(det.label).lower() != "person":
            continue
        boxes.append(tuple(float(x) for x in det.bbox_xyxy))
    return boxes
