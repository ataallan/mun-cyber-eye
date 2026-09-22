"""Combine sport, scene-place, kit, body-aggression, fall, gunshot, weapon, and face assists.

Single enrichment point so MOCK, YOLO, and the activity adapter share
the same metadata contract without double-counting cues.

Place type is a catalog setting (court / street / corridor / house / roam / …),
not recognition of a named arena. Kit cues are clothing-color clusters,
not identity or guilt. Fall manner is not a medical diagnosis. Gunshot
video proxies are not ballistic proof. Aimed-firearm geometry is not
proof of a real gun or intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from vision.aggression import (
    AggressionAssessment,
    analyze_aggression,
    detections_from_aggression,
)
from vision.detector import Detection
from vision.face_aggression import FaceAggressionResult, analyze_face_aggression
from vision.fall_assist import FallAssessment, analyze_fall_manner, person_boxes_from_detections
from vision.gunshot_assist import (
    GunshotAssessment,
    analyze_gunshot_proxy,
    detections_from_gunshot,
)
from vision.scene_context import (
    KitCues,
    PlaceAssessment,
    analyze_kit_cues,
    infer_scene_place,
    kit_sport_confidence_boost,
    place_display_name,
)
from vision.sports_catalog import infer_sport_context, resolve_sport, sport_display_name
from vision.improvised_hit import (
    ImprovisedHitAssessment,
    analyze_improvised_hit,
    apply_improvised_hit,
    detections_from_improvised_hit,
)
from vision.throw_assist import ThrowAssessment, analyze_throw, detections_from_throw, tracks_from_detections
from vision.weapon_assist import WeaponAssessment, analyze_weapon_use, detections_from_weapon


@dataclass
class AssistState:
    prev_bgr: Optional[np.ndarray] = None
    prev_motion: Optional[float] = None
    prev_person_boxes: list = field(default_factory=list)
    prev_object_tracks: list = field(default_factory=list)


@dataclass
class SceneAssist:
    sport_context: Optional[str] = None
    sport_confidence: float = 0.0
    sport_display: str = ""
    aggression: AggressionAssessment = field(default_factory=AggressionAssessment)
    face: FaceAggressionResult = field(default_factory=FaceAggressionResult)
    place: PlaceAssessment = field(default_factory=PlaceAssessment)
    kit: KitCues = field(default_factory=KitCues)
    fall: FallAssessment = field(default_factory=FallAssessment)
    gunshot: GunshotAssessment = field(default_factory=GunshotAssessment)
    weapon: WeaponAssessment = field(default_factory=WeaponAssessment)
    throw: ThrowAssessment = field(default_factory=ThrowAssessment)
    improvised_hit: ImprovisedHitAssessment = field(default_factory=ImprovisedHitAssessment)

    def to_extras(self) -> dict:
        extras: dict = {
            "aggression": self.aggression.to_dict(),
            "face_aggression": self.face.to_dict(),
            "scene_context": self.place.to_dict(),
            "kit": self.kit.to_dict(),
            "place_type": self.place.place_type,
            "place_display": self.place.display or place_display_name(self.place.place_type),
            "place_confidence": round(float(self.place.confidence), 3),
            "place_source": self.place.source,
            "team_kit_similarity": round(float(self.kit.team_kit_similarity), 3),
            "jersey_like_colors": bool(self.kit.jersey_like_colors),
            "fall": self.fall.to_dict(),
            "fall_manner": self.fall.manner,
            "fall_confidence": round(float(self.fall.confidence), 3),
            "gunshot": self.gunshot.to_dict(),
            "weapon": self.weapon.to_dict(),
            "weapon_use_intensity": round(float(self.weapon.use_intensity), 3),
            "weapon_use_tier": self.weapon.use_tier,
            "use_intensity_label": self.weapon.use_intensity_label,
            "weapon_class": self.weapon.weapon_class,
            "harm_potential": self.weapon.harm_potential,
            "weapon_id": self.weapon.weapon_id,
            "aimed_at_person": bool(self.weapon.aimed_at_person),
            "throw": self.throw.to_dict(),
            "thrown_at_person": bool(self.throw.thrown_at_person),
            "throw_confidence": round(float(self.throw.confidence), 3),
            "improvised_hit": bool(self.improvised_hit.reported),
            "improvised_object_label": (
                self.improvised_hit.object_label if self.improvised_hit.reported else ""
            ),
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


def _existing_place(detections: Sequence[Detection]) -> Optional[PlaceAssessment]:
    best: Optional[PlaceAssessment] = None
    best_conf = 0.0
    for det in detections:
        extras = det.extras or {}
        raw = extras.get("place_type")
        scene = extras.get("scene_context")
        if not raw and isinstance(scene, dict):
            raw = scene.get("place_type")
        if not raw:
            continue
        try:
            conf = float(
                extras.get("place_confidence")
                or (scene.get("place_confidence") if isinstance(scene, dict) else 0)
                or det.confidence
            )
        except (TypeError, ValueError):
            conf = float(det.confidence)
        if conf >= best_conf:
            best_conf = conf
            source = str(
                extras.get("place_source")
                or (scene.get("place_source") if isinstance(scene, dict) else "")
                or "detection"
            )
            best = PlaceAssessment(
                place_type=str(raw),
                confidence=conf,
                display=str(
                    extras.get("place_display")
                    or (scene.get("place_display") if isinstance(scene, dict) else "")
                    or place_display_name(str(raw))
                ),
                source=source,
                note=str(
                    extras.get("place_note")
                    or (scene.get("note") if isinstance(scene, dict) else "")
                    or ""
                ),
            )
    return best


def _catalog_object_ids(detections: Sequence[Detection]) -> list[str]:
    from vision.objects_catalog import map_detector_label

    ids: list[str] = []
    for det in detections:
        extras = det.extras or {}
        raw = extras.get("catalog_object_id") or det.label
        entry = map_detector_label(str(raw)) if raw else None
        if entry is not None:
            ids.append(entry.id)
    return ids


def _activity_labels(detections: Sequence[Detection]) -> set[str]:
    from vision.dataset import canonicalize_category

    found: set[str] = set()
    for det in detections:
        try:
            found.add(canonicalize_category(det.label))
        except ValueError:
            continue
    return found


_THREAT_ACTIVITY = {
    "potential_fight",
    "potential_fall",
    "potential_weapon_object",
}


def enrich_detections(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
    state: Optional[AssistState] = None,
    *,
    enable_face: Optional[bool] = None,
    camera_place_type: Optional[str] = None,
    location_label: Optional[str] = None,
    data_root: Optional[str | Path] = None,
) -> tuple[List[Detection], SceneAssist]:
    """Attach sport / place / kit / aggression / face metadata."""
    state = state or AssistState()
    dets = list(detections)
    sport_id, sport_conf = _existing_sport(dets)
    activity_labels = _activity_labels(dets)
    inferred_id, inferred_conf = infer_sport_context(image_bgr)
    # Never invent a sport on fight / fall / weapon frames — reddish
    # confrontation painters must not become "basketball."
    # Only infer a sport when the activity model already said game_or_play
    # (or MOCK/folder extras already named one). Color-wash frames must not
    # invent tennis/basketball from a tinted blank.
    if (
        sport_id is None
        and inferred_id
        and "game_or_play" in activity_labels
        and inferred_conf >= 0.50
        and not (activity_labels & _THREAT_ACTIVITY)
    ):
        sport_id, sport_conf = inferred_id, inferred_conf

    kit = analyze_kit_cues(image_bgr, dets)
    if sport_id:
        sport_conf = kit_sport_confidence_boost(kit, float(sport_conf or 0.0))

    # Camera stamp wins when set. Else extras already on the detection,
    # then labeled folders, then heuristic. Do not invent a sports venue
    # on threat-class frames from a color wash.
    allow_heuristic = not bool(activity_labels & _THREAT_ACTIVITY)
    object_ids = _catalog_object_ids(dets)
    if camera_place_type:
        place = infer_scene_place(
            image_bgr,
            camera_place_type=camera_place_type,
            data_root=data_root,
            allow_heuristic=False,
            object_ids=object_ids,
        )
    else:
        place = _existing_place(dets) or infer_scene_place(
            image_bgr,
            camera_place_type=None,
            data_root=data_root,
            allow_heuristic=allow_heuristic,
            object_ids=object_ids,
        )
    _ = location_label  # operator free-text only; never used as a famous arena

    aggression = analyze_aggression(
        image_bgr,
        prev_bgr=state.prev_bgr,
        detections=dets,
        prev_motion=state.prev_motion,
    )
    face = analyze_face_aggression(image_bgr, enabled=enable_face)
    fall = analyze_fall_manner(
        dets,
        aggression=aggression,
        prev_person_boxes=state.prev_person_boxes or None,
    )
    gunshot = analyze_gunshot_proxy(
        image_bgr,
        dets,
        aggression=aggression,
        prev_bgr=state.prev_bgr,
    )
    weapon = analyze_weapon_use(
        dets,
        aggression=aggression,
        prev_tracks=state.prev_object_tracks or None,
    )
    thrown = analyze_throw(dets, prev_tracks=state.prev_object_tracks or None)
    improvised = analyze_improvised_hit(
        dets,
        prev_tracks=state.prev_object_tracks or None,
    )
    weapon = apply_improvised_hit(weapon, improvised)

    assist = SceneAssist(
        sport_context=sport_id,
        sport_confidence=float(sport_conf or 0.0),
        sport_display=sport_display_name(sport_id) if sport_id else "",
        aggression=aggression,
        face=face,
        place=place,
        kit=kit,
        fall=fall,
        gunshot=gunshot,
        weapon=weapon,
        throw=thrown,
        improvised_hit=improvised,
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
                confidence=max(
                    assist.aggression.score,
                    assist.sport_confidence,
                    assist.place.confidence,
                    0.2,
                ),
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
    for extra in detections_from_gunshot(gunshot, bbox=bbox):
        if extra.label.lower() not in existing_labels:
            extra.extras.update(extras)
            merged.append(extra)
            existing_labels.add(extra.label.lower())
    for extra in detections_from_weapon(weapon, bbox=bbox):
        if extra.label.lower() not in existing_labels:
            extra.extras.update(extras)
            merged.append(extra)
            existing_labels.add(extra.label.lower())
    for extra in detections_from_throw(thrown, bbox=bbox):
        if extra.label.lower() not in existing_labels:
            extra.extras.update(extras)
            merged.append(extra)
            existing_labels.add(extra.label.lower())
    for extra in detections_from_improvised_hit(improvised, bbox=bbox):
        if extra.label.lower() not in existing_labels:
            extra.extras.update(extras)
            merged.append(extra)
            existing_labels.add(extra.label.lower())

    if image_bgr is not None and getattr(image_bgr, "size", 0) > 0:
        state.prev_bgr = np.array(image_bgr, copy=True)
        state.prev_motion = aggression.motion_intensity
        boxes = person_boxes_from_detections(dets)
        if boxes:
            state.prev_person_boxes = boxes
        tracks = tracks_from_detections(dets)
        if tracks:
            state.prev_object_tracks = tracks
    return merged, assist
