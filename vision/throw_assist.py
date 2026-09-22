"""Thrown-object-toward-person assist (CPU bbox proxies).

A small/medium object that translates quickly toward another person may
be a pass, a pitch, or an assault. Sport balls on a court are softened;
brick/bottle/improvised on a street/corridor/house are not. Distinct from
``firearm_aimed_at_person`` and from a static brandish. Humans verify.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from vision.detector import Detection

THROW_LABEL = "object_thrown_at_person"
USE_THROWN = "thrown_projectile"

SPORT_PROJECTILES = frozenset(
    {
        "sports ball",
        "sports_ball",
        "ball",
        "baseball",
        "soccer ball",
        "tennis ball",
        "frisbee",
        "football",
    }
)
HARMFUL_IMPROVISED = frozenset(
    {
        "bottle",
        "wine glass",
        "cup",
        "knife",
        "scissors",
        "baseball bat",
        "brick",
        "rock",
        "vase",
        "bowl",
        "laptop",
        "chair",
        "remote",
        "cell phone",
        "book",
        "handbag",
        "backpack",
        "suitcase",
        "umbrella",
        "suspicious_object",
        "raised_object",
    }
)
FIREARM_SKIP = frozenset({"gun", "firearm", "pistol", "rifle", "handgun"})
SKIP_LABELS = frozenset(
    {
        "person",
        "person_down",
        "horizontal_pose",
        "close_proximity",
        "rapid_motion",
        "strike_motion",
        "aggressive_pose",
        "aggressive_motion",
        "scene_assist",
        "ordinary",
        "game_or_play",
        "dance",
        "potential_fight",
        "potential_fall",
        "potential_weapon_object",
        "potential_gunshot",
        "possible_gunshot_video_proxy",
        "firearm_aimed_at_person",
        "weapon_pointed_at_person",
        "fall",
        "collapse",
        "unidentified_striking_object",
        "improvised_hit",
    }
) | FIREARM_SKIP

HONEST_NOTE = (
    "Thrown-object assist is a weak frame-to-frame bbox proxy — not proof "
    "of assault. Passes, pitches, and camera jitter look similar. Humans verify."
)
THROW_RATIONALE = (
    "Possible object thrown toward a person — verify. Not proof of assault."
)


@dataclass
class ObjectTrack:
    label: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]


@dataclass
class ThrowAssessment:
    thrown_at_person: bool = False
    confidence: float = 0.0
    use_tier: str = ""
    object_label: str = ""
    harmful: bool = False
    sport_projectile: bool = False
    cues: list[str] = field(default_factory=list)
    note: str = HONEST_NOTE

    def to_dict(self) -> dict:
        return {
            "thrown_at_person": bool(self.thrown_at_person),
            "confidence": round(float(self.confidence), 3),
            "use_tier": self.use_tier,
            "object_label": self.object_label,
            "harmful": bool(self.harmful),
            "sport_projectile": bool(self.sport_projectile),
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
        return (0.0, 0.0)
    return (vec[0] / mag, vec[1] / mag)


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _area(bbox: tuple[float, ...]) -> float:
    w, h = _wh(bbox)
    return w * h


def label_is_unidentified_object(label: str) -> bool:
    """True when a detector label is not a named catalog or dangerous class.

    Sport balls and assist labels are excluded. The unidentified improvised
    class itself stays in this set so a hit with that id can still be reviewed.
    """
    raw = (label or "").strip().lower()
    if not raw or raw in SPORT_PROJECTILES or raw in SKIP_LABELS or raw in FIREARM_SKIP:
        return False
    from vision.dangerous_objects import map_detector_label as map_dangerous
    from vision.objects_catalog import map_detector_label as map_object

    dangerous = map_dangerous(raw)
    if dangerous is not None and dangerous.id != "unidentified_improvised":
        return False
    if map_object(raw) is not None:
        return False
    return True


def tracks_from_detections(detections: Sequence[Detection]) -> list[ObjectTrack]:
    tracks: list[ObjectTrack] = []
    for det in detections:
        label = str(det.label).lower()
        if label in SKIP_LABELS:
            continue
        bbox = tuple(float(x) for x in det.bbox_xyxy)
        if _area(bbox) < 8.0:
            continue
        tracks.append(ObjectTrack(label=label, bbox=bbox, center=_center(bbox)))
    return tracks


def _match_prev(track: ObjectTrack, prev: Sequence[ObjectTrack]) -> Optional[ObjectTrack]:
    best = None
    best_d = 1e9
    for other in prev:
        if other.label != track.label and other.label not in SPORT_PROJECTILES | HARMFUL_IMPROVISED:
            # still allow same-class preference; otherwise nearest similar size
            if abs(_area(other.bbox) - _area(track.bbox)) / max(_area(track.bbox), 1.0) > 0.85:
                continue
        d = _dist(track.center, other.center)
        if d < best_d:
            best_d = d
            best = other
    if best is None or best_d > 220:
        return None
    return best


def _is_small_medium(obj: ObjectTrack, people: Sequence[Detection]) -> bool:
    area = _area(obj.bbox)
    if people:
        p_areas = [_area(p.bbox_xyxy) for p in people]
        median_p = sorted(p_areas)[len(p_areas) // 2]
        return 0.015 * median_p <= area <= 0.85 * median_p
    w, h = _wh(obj.bbox)
    return 6.0 <= max(w, h) <= 160.0


def analyze_throw(
    detections: Sequence[Detection],
    prev_tracks: Optional[Sequence[ObjectTrack]] = None,
) -> ThrowAssessment:
    """Detect a fast-moving small/medium object closing on another person."""
    people = [d for d in detections if str(d.label).lower() == "person"]
    objects = tracks_from_detections(detections)
    result = ThrowAssessment()
    if len(people) < 2 or not objects or not prev_tracks:
        return result

    best_conf = 0.0
    best: Optional[ThrowAssessment] = None
    for obj in objects:
        if not _is_small_medium(obj, people):
            continue
        prev = _match_prev(obj, prev_tracks)
        if prev is None:
            continue
        motion = _dist(obj.center, prev.center)
        scale = max(_wh(obj.bbox)[0], _wh(prev.bbox)[0], 8.0)
        if motion < 0.85 * scale:
            continue
        motion_dir = _norm((obj.center[0] - prev.center[0], obj.center[1] - prev.center[1]))
        if motion_dir == (0.0, 0.0):
            continue

        # Holder = person nearest to the *previous* location (release).
        holder = min(people, key=lambda p: _dist(_center(p.bbox_xyxy), prev.center))
        holder_c = _center(holder.bbox_xyxy)
        separated = _dist(obj.center, holder_c) > _dist(prev.center, holder_c) + 4.0

        toward = False
        closing = False
        for other in people:
            if other is holder:
                continue
            oc = _center(other.bbox_xyxy)
            to_other = _norm((oc[0] - obj.center[0], oc[1] - obj.center[1]))
            if _dot(motion_dir, to_other) < 0.55:
                continue
            d_now = _dist(obj.center, oc)
            d_prev = _dist(prev.center, oc)
            if d_now >= d_prev - 2.0:
                continue
            toward = True
            closing = True
            break
        if not toward:
            continue

        sport = obj.label in SPORT_PROJECTILES or prev.label in SPORT_PROJECTILES
        unidentified = label_is_unidentified_object(obj.label) or label_is_unidentified_object(
            prev.label
        )
        harmful = (
            obj.label in HARMFUL_IMPROVISED
            or prev.label in HARMFUL_IMPROVISED
            or unidentified
        )
        cues = ["fast_translation", "closing_on_person"]
        if unidentified:
            cues.append("unidentified_improvised")
        conf = 0.42 + min(0.22, motion / (scale * 8.0))
        if closing:
            conf += 0.08
        if separated:
            cues.append("release_from_holder")
            conf += 0.10
        conf = float(min(0.78, conf))
        assessment = ThrowAssessment(
            thrown_at_person=True,
            confidence=round(conf, 3),
            use_tier=USE_THROWN,
            object_label=obj.label,
            harmful=bool(harmful),
            sport_projectile=bool(sport),
            cues=cues,
            note=THROW_RATIONALE + " " + HONEST_NOTE,
        )
        if conf > best_conf:
            best_conf = conf
            best = assessment
    return best or result


def throw_should_soften(
    assessment: ThrowAssessment,
    *,
    sport_context: Optional[str] = None,
    sports_venue: bool = False,
    aimed_at_person: bool = False,
    aggression_high: bool = False,
    confrontation_setting: bool = False,
) -> bool:
    """Ball sport on a court/field stays play; street bottle/brick does not."""
    if not assessment.thrown_at_person:
        return False
    if aimed_at_person:
        return False
    if assessment.harmful and not assessment.sport_projectile:
        if confrontation_setting or not sport_context:
            return False
    if assessment.sport_projectile and sport_context:
        if aimed_at_person and aggression_high:
            return False
        return True
    if sport_context and sports_venue and not assessment.harmful:
        return True
    return False


def detections_from_throw(
    assessment: ThrowAssessment,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list[Detection]:
    if not assessment.thrown_at_person:
        return []
    return [
        Detection(
            label=THROW_LABEL,
            confidence=float(assessment.confidence),
            bbox_xyxy=bbox,
            extras={"throw": assessment.to_dict(), "use_tier": USE_THROWN},
        )
    ]
