"""Dangerous-object use intensity and aimed-firearm geometry.

These are **weak OpenCV / bbox proxies**. A phone, toy, or occlusion can
look like a pointed gun. Sport context may soften a bat on a field; it
must **not** suppress ``firearm_aimed_at_person``. Humans verify. No
enforcement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from vision.aggression import AggressionAssessment
from vision.detector import Detection

FIREARM_LABELS = frozenset(
    {"gun", "firearm", "pistol", "rifle", "handgun"}
)
# Elongated weapon-like proxies that may be aimed (not sports equipment).
AIMABLE_LABELS = FIREARM_LABELS | {"knife", "raised_object"}
SPORT_SOFTENABLE = frozenset({"baseball bat", "sports ball", "scissors"})

USE_PRESENCE = "presence"
USE_BRANDISH = "brandish"
USE_THREATENING = "threatening_motion"
USE_AIMED = "aimed_at_person"

HONEST_NOTE = (
    "Weapon-like object and aim direction are bbox proxies — not proof of a "
    "real firearm, a loaded weapon, or intent to harm. Toys, phones, and "
    "occlusion fool this assist. Humans verify. No enforcement."
)
AIMED_RATIONALE = (
    "Possible firearm-like object pointed toward a person — verify. "
    "Not proof of a real firearm or intent."
)


@dataclass
class WeaponAssessment:
    present: bool = False
    labels: list[str] = field(default_factory=list)
    use_intensity: float = 0.0
    use_tier: str = ""
    aimed_at_person: bool = False
    aim_confidence: float = 0.0
    cues: list[str] = field(default_factory=list)
    note: str = HONEST_NOTE

    def to_dict(self) -> dict:
        return {
            "present": bool(self.present),
            "labels": list(self.labels),
            "use_intensity": round(float(self.use_intensity), 3),
            "use_tier": self.use_tier,
            "aimed_at_person": bool(self.aimed_at_person),
            "aim_confidence": round(float(self.aim_confidence), 3),
            "cues": list(self.cues),
            "note": self.note,
        }


def _center(bbox: tuple[float, ...]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _wh(bbox: tuple[float, ...]) -> tuple[float, float]:
    return (
        max(1.0, abs(float(bbox[2]) - float(bbox[0]))),
        max(1.0, abs(float(bbox[3]) - float(bbox[1]))),
    )


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _norm(vec: tuple[float, float]) -> tuple[float, float]:
    mag = math.hypot(vec[0], vec[1])
    if mag < 1e-6:
        return (1.0, 0.0)
    return (vec[0] / mag, vec[1] / mag)


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _long_axis_tip(
    bbox: tuple[float, ...], holder: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Tip = bbox endpoint along the long axis that is farther from the holder."""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    w, h = _wh(bbox)
    cx, cy = _center(bbox)
    if h >= w:
        ends = ((cx, y1), (cx, y2))
    else:
        ends = ((x1, cy), (x2, cy))
    tip = max(ends, key=lambda p: _dist(p, holder))
    direction = _norm((tip[0] - holder[0], tip[1] - holder[1]))
    return tip, direction


def _ray_points_at_bbox(
    origin: tuple[float, float],
    direction: tuple[float, float],
    bbox: tuple[float, ...],
    *,
    cone_cos: float = 0.72,
    max_range_scale: float = 10.0,
) -> bool:
    pc = _center(bbox)
    vec = (pc[0] - origin[0], pc[1] - origin[1])
    dist = math.hypot(vec[0], vec[1])
    if dist < 4.0:
        return False
    unit = _norm(vec)
    if _dot(direction, unit) < cone_cos:
        return False
    pw, ph = _wh(bbox)
    scale = max(pw, ph, 8.0)
    if dist > scale * max_range_scale:
        return False
    return True


def analyze_weapon_use(
    detections: Sequence[Detection],
    aggression: Optional[AggressionAssessment] = None,
) -> WeaponAssessment:
    """Presence / brandish / threatening motion / aimed-at-person."""
    aggression = aggression or AggressionAssessment()
    people = [d for d in detections if str(d.label).lower() == "person"]
    weapons = [
        d
        for d in detections
        if str(d.label).lower() in AIMABLE_LABELS
        or str(d.label).lower() in FIREARM_LABELS
    ]
    # YOLO may emit "raised_object" next to a knife; keep firearm-like + aimable.
    extra_weaponish = [
        d
        for d in detections
        if str(d.label).lower() in {"suspicious_object"}
        and str((d.extras or {}).get("source_label") or "").lower() in FIREARM_LABELS
    ]
    weapons = weapons + extra_weaponish
    labels = sorted({str(d.label).lower() for d in weapons})
    result = WeaponAssessment(present=bool(weapons), labels=labels)
    if not weapons:
        return result

    cues = ["weapon_like_object"]
    intensity = 0.34  # presence
    tier = USE_PRESENCE
    raised = any(str(d.label).lower() == "raised_object" for d in detections)
    firearm = any(str(d.label).lower() in FIREARM_LABELS for d in weapons)
    if raised or firearm:
        intensity = max(intensity, 0.55)
        tier = USE_BRANDISH
        cues.append("brandish")
    if float(aggression.score) >= 0.45 or float(aggression.raised_arm) >= 0.12:
        intensity = max(intensity, 0.72)
        tier = USE_THREATENING
        cues.append("threatening_motion")

    aimed = False
    aim_conf = 0.0
    if len(people) >= 2 and weapons:
        for weapon in weapons:
            holder = min(people, key=lambda p: _dist(_center(p.bbox_xyxy), _center(weapon.bbox_xyxy)))
            holder_c = _center(holder.bbox_xyxy)
            tip, direction = _long_axis_tip(weapon.bbox_xyxy, holder_c)
            for other in people:
                if other is holder:
                    continue
                if _ray_points_at_bbox(tip, direction, other.bbox_xyxy):
                    aimed = True
                    aim_conf = max(aim_conf, 0.70)
                    break
            if aimed:
                break
        if aimed:
            cues.append("firearm_aimed_at_person")
            intensity = max(intensity, 0.92)
            tier = USE_AIMED
    elif len(people) <= 1:
        cues.append("brandish_only")

    result.use_intensity = round(float(min(1.0, intensity)), 3)
    result.use_tier = tier
    result.aimed_at_person = bool(aimed)
    result.aim_confidence = round(float(aim_conf), 3)
    result.cues = cues
    if aimed:
        result.note = AIMED_RATIONALE + " " + HONEST_NOTE
    return result


def detections_from_weapon(
    assessment: WeaponAssessment,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list[Detection]:
    extra: list[Detection] = []
    if assessment.aimed_at_person:
        extra.append(
            Detection(
                label="firearm_aimed_at_person",
                confidence=max(0.7, float(assessment.aim_confidence) or 0.7),
                bbox_xyxy=bbox,
                extras={"weapon": assessment.to_dict(), "use_tier": USE_AIMED},
            )
        )
    return extra
