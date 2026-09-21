"""Dangerous-object use intensity and aimed-firearm geometry.

These are **weak OpenCV / bbox proxies**. A phone, toy, tool, or
occlusion can look like a weapon. Sport context may soften a bat on a
field at low use-intensity; it must **not** suppress
``firearm_aimed_at_person``. Indicator only — not proof of assault or
intent. Humans verify. No enforcement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from vision.aggression import AggressionAssessment
from vision.dangerous_objects import (
    CLASS_FIREARM,
    HONEST_NOTE,
    USE_BRANDISHED,
    USE_NONE,
    USE_STRIKE,
    USE_THREATENING,
    DangerousEntry,
    map_detector_label,
)
from vision.detector import Detection

USE_AIMED = "aimed_at_person"

AIMED_RATIONALE = (
    "Possible firearm-like object pointed toward a person — verify. "
    "Not proof of a real firearm or intent."
)
INTENSITY_NOTE = (
    "Use-against-person intensity is an indicator only — not proof of "
    "assault or intent. Toys, tools, and phones false-fire."
)


@dataclass
class WeaponAssessment:
    present: bool = False
    labels: list[str] = field(default_factory=list)
    weapon_id: str = ""
    weapon_class: str = ""
    harm_potential: str = ""
    use_intensity: float = 0.0
    use_intensity_label: str = USE_NONE
    use_tier: str = ""
    aimed_at_person: bool = False
    aim_confidence: float = 0.0
    toward_person: bool = False
    cues: list[str] = field(default_factory=list)
    note: str = HONEST_NOTE

    def to_dict(self) -> dict:
        return {
            "present": bool(self.present),
            "labels": list(self.labels),
            "weapon_id": self.weapon_id,
            "weapon_class": self.weapon_class,
            "harm_potential": self.harm_potential,
            "use_intensity": round(float(self.use_intensity), 3),
            "use_intensity_label": self.use_intensity_label or USE_NONE,
            "use_tier": self.use_tier,
            "aimed_at_person": bool(self.aimed_at_person),
            "aim_confidence": round(float(self.aim_confidence), 3),
            "toward_person": bool(self.toward_person),
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


def _norm_vec(vec: tuple[float, float]) -> tuple[float, float]:
    mag = math.hypot(vec[0], vec[1])
    if mag < 1e-6:
        return (1.0, 0.0)
    return (vec[0] / mag, vec[1] / mag)


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _long_axis_tip(
    bbox: tuple[float, ...], holder: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    w, h = _wh(bbox)
    cx, cy = _center(bbox)
    if h >= w:
        ends = ((cx, y1), (cx, y2))
    else:
        ends = ((x1, cy), (x2, cy))
    tip = max(ends, key=lambda p: _dist(p, holder))
    direction = _norm_vec((tip[0] - holder[0], tip[1] - holder[1]))
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
    unit = _norm_vec(vec)
    if _dot(direction, unit) < cone_cos:
        return False
    pw, ph = _wh(bbox)
    scale = max(pw, ph, 8.0)
    if dist > scale * max_range_scale:
        return False
    return True


def _entry_for(det: Detection) -> Optional[DangerousEntry]:
    extras = det.extras or {}
    raw = extras.get("dangerous_object_id") or extras.get("source_label") or det.label
    entry = map_detector_label(str(raw))
    if entry is not None:
        return entry
    label = str(det.label).lower()
    if label == "chair":
        return map_detector_label("chair_as_weapon")
    if label in {"raised_object", "suspicious_object"}:
        src = str(extras.get("source_label") or "")
        mapped = map_detector_label(src) if src else None
        return mapped or map_detector_label("heavy_stick")
    return None


def _is_raised(det: Detection, holder: Optional[Detection]) -> bool:
    if str(det.label).lower() == "raised_object":
        return True
    if holder is None:
        return False
    obj_c = _center(det.bbox_xyxy)
    _hx1, hy1, _hx2, hy2 = holder.bbox_xyxy
    top = min(float(hy1), float(hy2))
    height = max(8.0, abs(float(hy2) - float(hy1)))
    return obj_c[1] <= top + 0.38 * height


def analyze_weapon_use(
    detections: Sequence[Detection],
    aggression: Optional[AggressionAssessment] = None,
    prev_tracks: Optional[Sequence] = None,
) -> WeaponAssessment:
    """Presence / brandished / threatening_motion / possible_strike / aimed."""
    aggression = aggression or AggressionAssessment()
    people = [d for d in detections if str(d.label).lower() == "person"]
    labels_all = {str(d.label).lower() for d in detections}
    strike_label = "strike_motion" in labels_all or "aggressive_pose" in labels_all

    candidates: list[tuple[Detection, DangerousEntry]] = []
    for det in detections:
        entry = _entry_for(det)
        if entry is None:
            continue
        candidates.append((det, entry))

    result = WeaponAssessment(
        labels=sorted({e.id for _d, e in candidates} | (labels_all & {"raised_object"}))
    )
    if not candidates:
        return result

    closing = False
    if prev_tracks and people:
        from vision.throw_assist import ObjectTrack, _match_prev, tracks_from_detections

        tracks = tracks_from_detections([d for d, _e in candidates])
        prev_list = list(prev_tracks)
        if tracks and prev_list and isinstance(prev_list[0], ObjectTrack):
            for track in tracks:
                prev = _match_prev(track, prev_list)
                if prev is None:
                    continue
                holder = min(
                    people, key=lambda p: _dist(_center(p.bbox_xyxy), prev.center)
                )
                for other in people:
                    if other is holder:
                        continue
                    if _dist(track.center, _center(other.bbox_xyxy)) + 4.0 < _dist(
                        prev.center, _center(other.bbox_xyxy)
                    ):
                        closing = True
                        break

    best_intensity = -1.0
    best: Optional[tuple[DangerousEntry, dict]] = None
    for det, entry in candidates:
        holder = None
        if people:
            holder = min(
                people, key=lambda p: _dist(_center(p.bbox_xyxy), _center(det.bbox_xyxy))
            )
        raised = _is_raised(det, holder)
        toward = False
        aimed = False
        aim_conf = 0.0
        if holder is not None and len(people) >= 2:
            holder_c = _center(holder.bbox_xyxy)
            tip, direction = _long_axis_tip(det.bbox_xyxy, holder_c)
            obj_c = _center(det.bbox_xyxy)
            for other in people:
                if other is holder:
                    continue
                if _ray_points_at_bbox(tip, direction, other.bbox_xyxy):
                    toward = True
                    if entry.weapon_class == CLASS_FIREARM:
                        aimed = True
                        aim_conf = 0.70
                    break
                other_c = _center(other.bbox_xyxy)
                other_span = max(_wh(other.bbox_xyxy))
                # Nearby to another person only if the object is closer to
                # them than to the holder — two players on a field with a
                # bat at the holder's hip is not "toward".
                if (
                    _dist(obj_c, other_c) < 0.42 * other_span
                    and _dist(obj_c, other_c) + 8.0 < _dist(obj_c, holder_c)
                ):
                    toward = True

        strike_like = (
            strike_label
            or (
                float(aggression.raised_arm) >= 0.18
                and float(aggression.score) >= 0.55
            )
        ) and (toward or closing or len(people) >= 2)

        cues = ["dangerous_object"]
        intensity = 0.0
        label = USE_NONE
        if entry.context_dependent and not (
            raised or toward or closing or strike_like or aimed
        ):
            continue

        if not entry.context_dependent:
            intensity = 0.22
            label = USE_NONE
            cues.append("weapon_like_present")
        if raised or entry.weapon_class == CLASS_FIREARM:
            intensity = max(intensity, 0.48)
            label = USE_BRANDISHED
            cues.append("brandished")
        if toward or (closing and float(aggression.score) >= 0.40):
            intensity = max(intensity, 0.68)
            label = USE_THREATENING
            cues.append("toward_person")
        if closing:
            cues.append("closing_distance")
        if strike_label and len(people) >= 2:
            intensity = max(intensity, 0.68)
            if label == USE_NONE:
                label = USE_THREATENING
            cues.append("strike_motion_nearby")
        if strike_like and (toward or closing or raised):
            intensity = max(intensity, 0.86)
            label = USE_STRIKE
            cues.append("possible_strike")
        if aimed:
            intensity = max(intensity, 0.92)
            cues.append("firearm_aimed_at_person")
        if len(people) <= 1:
            cues.append("brandish_only")

        payload = {
            "intensity": intensity,
            "label": label,
            "aimed": aimed,
            "aim_conf": aim_conf,
            "toward": toward,
            "cues": cues,
        }
        rank = intensity + (0.05 if entry.harm_potential == "high" else 0.0)
        if rank > best_intensity:
            best_intensity = rank
            best = (entry, payload)

    if best is None:
        return result

    entry, payload = best
    result.present = True
    result.weapon_id = entry.id
    result.weapon_class = entry.weapon_class
    result.harm_potential = entry.harm_potential
    result.use_intensity = round(float(min(1.0, payload["intensity"])), 3)
    result.use_intensity_label = str(payload["label"] or USE_NONE)
    result.use_tier = USE_AIMED if payload["aimed"] else result.use_intensity_label
    result.aimed_at_person = bool(payload["aimed"])
    result.aim_confidence = round(float(payload["aim_conf"] or 0.0), 3)
    result.toward_person = bool(payload["toward"])
    result.cues = list(payload["cues"])
    result.labels = sorted(set(result.labels) | {entry.id})
    result.note = (
        (AIMED_RATIONALE + " " if result.aimed_at_person else "")
        + INTENSITY_NOTE
        + " "
        + HONEST_NOTE
    )
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
