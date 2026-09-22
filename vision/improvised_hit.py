"""Unidentified object used to hit another person.

Known edged, blunt, and firearm-like ids stay on the dangerous-object path.
This assist covers a detector box that is **not** in the home catalog and
**not** a named dangerous class, when that box moves into contact with a
second person. A single still frame, a dead stream, or an object that is
not closing does not produce a cue. Humans verify. No enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from vision.dangerous_objects import CLASS_IMPROVISED, HARM_HIGH, USE_STRIKE
from vision.detector import Detection
from vision.throw_assist import (
    ObjectTrack,
    _center,
    _dist,
    _match_prev,
    _wh,
    label_is_unidentified_object,
    tracks_from_detections,
)

HIT_LABEL = "unidentified_striking_object"

HONEST_NOTE = (
    "Improvised / unidentified striking-object assist is a weak bbox proxy — "
    "not proof of assault or of a weapon. Humans verify. The system does not enforce."
)
HIT_RATIONALE = (
    "Possible improvised / unidentified object used to hit another person — "
    "verify. Not a named weapon. Not proof of assault or intent."
)


@dataclass
class ImprovisedHitAssessment:
    hit: bool = False
    reported: bool = False
    confidence: float = 0.0
    object_label: str = ""
    cues: list[str] = field(default_factory=list)
    note: str = HONEST_NOTE

    def to_dict(self) -> dict:
        return {
            "hit": bool(self.hit),
            "reported": bool(self.reported),
            "confidence": round(float(self.confidence), 3),
            "object_label": self.object_label,
            "cues": list(self.cues),
            "note": self.note,
        }


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    ax1, ay1, ax2, ay2 = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bx1, by1, bx2, by2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _plausible_size(obj: ObjectTrack, people: Sequence[Detection]) -> bool:
    w, h = _wh(obj.bbox)
    area = w * h
    if area < 12.0 or max(w, h) < 6.0:
        return False
    if not people:
        return False
    areas = []
    for person in people:
        pw, ph = _wh(person.bbox_xyxy)
        areas.append(pw * ph)
    median = sorted(areas)[len(areas) // 2]
    if median <= 1.0:
        return False
    return 0.01 * median <= area <= 1.15 * median


def analyze_improvised_hit(
    detections: Sequence[Detection],
    prev_tracks: Optional[Sequence[ObjectTrack]] = None,
) -> ImprovisedHitAssessment:
    """Contact strike by an object that is not a named catalog class.

    Requires two people, a previous object position, and motion into the
    other person. Returns a negative assessment when those are missing.
    """
    people = [d for d in detections if str(d.label).lower() == "person"]
    result = ImprovisedHitAssessment()
    if len(people) < 2 or not prev_tracks:
        return result

    best: Optional[ImprovisedHitAssessment] = None
    best_conf = 0.0
    for obj in tracks_from_detections(detections):
        if not label_is_unidentified_object(obj.label):
            continue
        if not _plausible_size(obj, people):
            continue
        prev = _match_prev(obj, prev_tracks)
        if prev is None or not label_is_unidentified_object(prev.label):
            continue
        motion = _dist(obj.center, prev.center)
        scale = max(_wh(obj.bbox)[0], _wh(prev.bbox)[0], 8.0)
        if motion < 0.45 * scale:
            continue
        holder = min(people, key=lambda p: _dist(_center(p.bbox_xyxy), prev.center))
        for other in people:
            if other is holder:
                continue
            other_c = _center(other.bbox_xyxy)
            d_now = _dist(obj.center, other_c)
            d_prev = _dist(prev.center, other_c)
            if d_now + 2.0 >= d_prev:
                continue
            span = max(_wh(other.bbox_xyxy))
            touching = _overlap(obj.bbox, other.bbox_xyxy) or d_now <= 0.28 * span
            if not touching:
                continue
            conf = 0.62 + min(0.10, motion / (scale * 12.0))
            if _overlap(obj.bbox, other.bbox_xyxy):
                conf += 0.06
            conf = float(min(0.78, conf))
            assessment = ImprovisedHitAssessment(
                hit=True,
                confidence=round(conf, 3),
                object_label=obj.label,
                cues=["closing_on_person", "contact", "unidentified_improvised"],
                note=HIT_RATIONALE + " " + HONEST_NOTE,
            )
            if conf > best_conf:
                best_conf = conf
                best = assessment
    return best or result


def apply_improvised_hit(weapon, assessment: ImprovisedHitAssessment):
    """Fold a hit into weapon metadata unless a named weapon is already in use.

    A knife or bat that is already at threatening / strike intensity keeps
    its own id. An idle named object does not hide an unidentified hit.
    """
    assessment.reported = False
    if weapon is None or not assessment.hit:
        return weapon
    known_active = (
        bool(getattr(weapon, "present", False))
        and bool(getattr(weapon, "weapon_id", ""))
        and str(weapon.weapon_id) != "unidentified_improvised"
        and float(getattr(weapon, "use_intensity", 0) or 0) >= 0.68
    )
    if known_active:
        return weapon
    assessment.reported = True
    weapon.present = True
    weapon.weapon_id = "unidentified_improvised"
    weapon.weapon_class = CLASS_IMPROVISED
    weapon.harm_potential = HARM_HIGH
    weapon.use_intensity = round(max(float(weapon.use_intensity or 0), 0.86), 3)
    weapon.use_intensity_label = USE_STRIKE
    weapon.use_tier = USE_STRIKE
    weapon.toward_person = True
    cues = list(weapon.cues or [])
    for cue in ("unidentified_striking_object", "possible_strike", "improvised_hit"):
        if cue not in cues:
            cues.append(cue)
    weapon.cues = cues
    labels = set(weapon.labels or [])
    labels.add("unidentified_improvised")
    weapon.labels = sorted(labels)
    weapon.note = HIT_RATIONALE + " " + HONEST_NOTE
    return weapon


def detections_from_improvised_hit(
    assessment: ImprovisedHitAssessment,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list[Detection]:
    if not assessment.reported:
        return []
    return [
        Detection(
            label=HIT_LABEL,
            confidence=float(assessment.confidence or 0.62),
            bbox_xyxy=bbox,
            extras={
                "improvised_hit": True,
                "improvised_object_label": assessment.object_label,
                "weapon_id": "unidentified_improvised",
                "use_tier": USE_STRIKE,
            },
        )
    ]
