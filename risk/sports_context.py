"""Sport-versus-confrontation policy helpers.

When ``sport_context`` is present with decent confidence, fight heuristics
are softened unless body-aggression cues are strong. High aggression during
a named sport stays ``game_or_play`` (possible intense play) unless
``ALERT_ON_INTENSE_SPORT=1``.

Place / venue type and kit-color similarity are additional assists:
sports venues strengthen play softening; street / corridor / house /
compound plus high aggression and no sport lean toward a fight alert.
Street play (street + sport) stays ``game_or_play`` unless aggression is
extreme.

This is assistive policy. It does not prove play is safe or that a scene
is a crime, and it never labels a face or a uniform as criminal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from vision.detector import Detection
from vision.scene_context import (
    KitCues,
    is_confrontation_setting,
    is_sports_venue,
    is_street_place,
    is_strong_confrontation_setting,
    kit_sport_confidence_boost,
    place_display_name,
    resolve_place,
)
from vision.sports_catalog import resolve_sport, sport_display_name

SPORT_CONFIDENCE_DECENT = 0.55
SPORT_CONFIDENCE_STRONG = 0.70
SPORT_CONFIDENCE_VENUE_ASSIST = 0.45
AGGRESSION_HIGH = 0.65
AGGRESSION_VERY_HIGH = 0.85
KIT_SIMILARITY_HELPFUL = 0.55

FIGHT_PROXY_LABELS = {
    "close_proximity",
    "rapid_motion",
    "strike_motion",
    "aggressive_pose",
    "aggressive_motion",
}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def alert_on_intense_sport(override: Optional[bool] = None) -> bool:
    if override is not None:
        return bool(override)
    return _env_flag("ALERT_ON_INTENSE_SPORT", "0")


@dataclass
class SceneContext:
    sport_context: Optional[str] = None
    sport_confidence: float = 0.0
    sport_display: str = ""
    aggression_score: float = 0.0
    aggression_cues: List[str] = field(default_factory=list)
    face_status: str = "disabled"
    face_note: str = ""
    face_score: Optional[float] = None
    place_type: str = "unknown"
    place_confidence: float = 0.0
    place_display: str = ""
    place_source: str = "none"
    place_note: str = ""
    team_kit_similarity: float = 0.0
    jersey_like_colors: bool = False
    kit_note: str = ""
    fall_manner: str = ""
    fall_confidence: float = 0.0
    fall_display: str = ""
    gunshot_proxy: bool = False
    gunshot_confidence: float = 0.0
    gunshot_audio_status: str = "disabled"
    aimed_at_person: bool = False
    weapon_use_intensity: float = 0.0
    weapon_use_tier: str = ""
    use_intensity_label: str = ""
    weapon_class: str = ""
    harm_potential: str = ""
    weapon_id: str = ""
    weapon_cues: List[str] = field(default_factory=list)
    thrown_at_person: bool = False
    throw_confidence: float = 0.0
    throw_label: str = ""
    throw_harmful: bool = False
    throw_sport_projectile: bool = False
    throw_cues: List[str] = field(default_factory=list)

    @property
    def sports_venue(self) -> bool:
        return is_sports_venue(self.place_type) and self.place_confidence >= 0.50

    @property
    def confrontation_setting(self) -> bool:
        return is_confrontation_setting(self.place_type) and self.place_confidence >= 0.50

    @property
    def strong_confrontation_setting(self) -> bool:
        return (
            is_strong_confrontation_setting(self.place_type)
            and self.place_confidence >= 0.50
        )

    @property
    def street_setting(self) -> bool:
        return is_street_place(self.place_type) and self.place_confidence >= 0.50

    @property
    def kit_supports_play(self) -> bool:
        return self.team_kit_similarity >= KIT_SIMILARITY_HELPFUL

    @property
    def sport_decent(self) -> bool:
        if not self.sport_context:
            return False
        threshold = (
            SPORT_CONFIDENCE_VENUE_ASSIST
            if self.sports_venue
            else SPORT_CONFIDENCE_DECENT
        )
        return self.sport_confidence >= threshold

    @property
    def sport_strong(self) -> bool:
        if not self.sport_context:
            return False
        # Sports venue + named sport is a stronger play signal.
        if self.sports_venue and self.sport_confidence >= SPORT_CONFIDENCE_DECENT:
            return True
        return self.sport_confidence >= SPORT_CONFIDENCE_STRONG

    @property
    def aggression_high(self) -> bool:
        return self.aggression_score >= AGGRESSION_HIGH

    @property
    def aggression_very_high(self) -> bool:
        return self.aggression_score >= AGGRESSION_VERY_HIGH


def collect_scene_context(detections: Sequence[Detection]) -> SceneContext:
    """Read sport / place / kit / aggression / face extras on detections."""
    ctx = SceneContext()
    cues: list[str] = []
    for det in detections:
        extras = det.extras or {}
        raw = extras.get("sport_context") or extras.get("sport")
        if raw:
            entry = resolve_sport(str(raw))
            sport_id = entry.id if entry is not None else str(raw).strip().lower()
            try:
                conf = float(extras.get("sport_confidence") or det.confidence)
            except (TypeError, ValueError):
                conf = float(det.confidence)
            if sport_id and conf >= ctx.sport_confidence:
                ctx.sport_context = sport_id
                ctx.sport_confidence = conf
                ctx.sport_display = str(
                    extras.get("sport_display") or sport_display_name(sport_id)
                )

        scene = extras.get("scene_context") if isinstance(extras.get("scene_context"), dict) else extras
        raw_place = extras.get("place_type") or (scene.get("place_type") if isinstance(scene, dict) else None)
        if raw_place:
            entry = resolve_place(str(raw_place))
            place_id = entry.id if entry is not None else str(raw_place).strip().lower()
            try:
                pconf = float(
                    extras.get("place_confidence")
                    or (scene.get("place_confidence") if isinstance(scene, dict) else 0)
                    or det.confidence
                )
            except (TypeError, ValueError):
                pconf = float(det.confidence)
            if place_id and pconf >= ctx.place_confidence:
                ctx.place_type = place_id
                ctx.place_confidence = pconf
                ctx.place_display = str(
                    extras.get("place_display")
                    or (scene.get("place_display") if isinstance(scene, dict) else "")
                    or place_display_name(place_id)
                )
                ctx.place_source = str(
                    extras.get("place_source")
                    or (scene.get("place_source") if isinstance(scene, dict) else "")
                    or ctx.place_source
                )
                ctx.place_note = str(
                    extras.get("place_note")
                    or (scene.get("note") if isinstance(scene, dict) else "")
                    or ctx.place_note
                )

        kit = extras.get("kit") if isinstance(extras.get("kit"), dict) else extras
        if isinstance(kit, dict):
            try:
                ctx.team_kit_similarity = max(
                    ctx.team_kit_similarity,
                    float(kit.get("team_kit_similarity") or extras.get("team_kit_similarity") or 0.0),
                )
            except (TypeError, ValueError):
                pass
            if kit.get("jersey_like_colors") or extras.get("jersey_like_colors"):
                ctx.jersey_like_colors = True
            ctx.kit_note = str(kit.get("note") or extras.get("kit_note") or ctx.kit_note)

        ag = extras.get("aggression")
        if isinstance(ag, dict):
            try:
                ctx.aggression_score = max(
                    ctx.aggression_score, float(ag.get("score") or 0.0)
                )
            except (TypeError, ValueError):
                pass
            extra_cues = ag.get("cues") or []
            if isinstance(extra_cues, (list, tuple)):
                cues.extend(str(c) for c in extra_cues)

        face = extras.get("face_aggression")
        if isinstance(face, dict):
            ctx.face_status = str(face.get("status") or face.get("face_aggression") or ctx.face_status)
            ctx.face_note = str(face.get("note") or ctx.face_note)
            if face.get("score") is not None:
                try:
                    ctx.face_score = float(face["score"])
                except (TypeError, ValueError):
                    ctx.face_score = ctx.face_score

        if det.label.lower() in FIGHT_PROXY_LABELS:
            cues.append(det.label.lower())
            # Heuristic label confidence is not a body-aggression score.
            # Using it here would make sport-softening impossible whenever
            # close_proximity / rapid_motion are present.

        fall = extras.get("fall") if isinstance(extras.get("fall"), dict) else extras
        raw_manner = extras.get("fall_manner") or (
            fall.get("fall_manner") if isinstance(fall, dict) else ""
        )
        if raw_manner:
            ctx.fall_manner = str(raw_manner)
            try:
                ctx.fall_confidence = max(
                    ctx.fall_confidence,
                    float(
                        extras.get("fall_confidence")
                        or (fall.get("fall_confidence") if isinstance(fall, dict) else 0)
                        or 0
                    ),
                )
            except (TypeError, ValueError):
                pass
            ctx.fall_display = str(
                extras.get("fall_display")
                or (fall.get("fall_display") if isinstance(fall, dict) else "")
                or ctx.fall_display
            )

        gunshot = extras.get("gunshot") if isinstance(extras.get("gunshot"), dict) else extras
        if extras.get("event_type") == "possible_gunshot" or str(det.label).lower() == "possible_gunshot_video_proxy":
            ctx.gunshot_proxy = True
        if isinstance(gunshot, dict) and gunshot.get("video_proxy"):
            ctx.gunshot_proxy = True
            try:
                ctx.gunshot_confidence = max(
                    ctx.gunshot_confidence, float(gunshot.get("confidence") or 0)
                )
            except (TypeError, ValueError):
                pass
        if isinstance(gunshot, dict) and gunshot.get("audio_status"):
            ctx.gunshot_audio_status = str(gunshot.get("audio_status"))

        weapon = extras.get("weapon") if isinstance(extras.get("weapon"), dict) else extras
        if extras.get("aimed_at_person") or str(det.label).lower() in {
            "firearm_aimed_at_person",
            "weapon_pointed_at_person",
        }:
            ctx.aimed_at_person = True
        if isinstance(weapon, dict):
            if weapon.get("aimed_at_person"):
                ctx.aimed_at_person = True
            try:
                ctx.weapon_use_intensity = max(
                    ctx.weapon_use_intensity,
                    float(weapon.get("use_intensity") or extras.get("weapon_use_intensity") or 0),
                )
            except (TypeError, ValueError):
                pass
            ctx.weapon_use_tier = str(
                weapon.get("use_tier") or extras.get("weapon_use_tier") or ctx.weapon_use_tier
            )
            ctx.use_intensity_label = str(
                weapon.get("use_intensity_label")
                or extras.get("use_intensity_label")
                or ctx.use_intensity_label
            )
            ctx.weapon_class = str(
                weapon.get("weapon_class") or extras.get("weapon_class") or ctx.weapon_class
            )
            ctx.harm_potential = str(
                weapon.get("harm_potential") or extras.get("harm_potential") or ctx.harm_potential
            )
            ctx.weapon_id = str(
                weapon.get("weapon_id") or extras.get("weapon_id") or ctx.weapon_id
            )
            extra_wc = weapon.get("cues") or []
            if isinstance(extra_wc, (list, tuple)):
                ctx.weapon_cues.extend(str(c) for c in extra_wc)

        thrown = extras.get("throw") if isinstance(extras.get("throw"), dict) else extras
        if extras.get("thrown_at_person") or str(det.label).lower() == "object_thrown_at_person":
            ctx.thrown_at_person = True
        if isinstance(thrown, dict) and thrown.get("thrown_at_person"):
            ctx.thrown_at_person = True
            try:
                ctx.throw_confidence = max(
                    ctx.throw_confidence, float(thrown.get("confidence") or 0)
                )
            except (TypeError, ValueError):
                pass
            ctx.throw_label = str(thrown.get("object_label") or ctx.throw_label)
            ctx.throw_harmful = bool(thrown.get("harmful") or ctx.throw_harmful)
            ctx.throw_sport_projectile = bool(
                thrown.get("sport_projectile") or ctx.throw_sport_projectile
            )
            extra_tc = thrown.get("cues") or []
            if isinstance(extra_tc, (list, tuple)):
                ctx.throw_cues.extend(str(c) for c in extra_tc)

    # de-dupe cues, keep order
    seen: set[str] = set()
    ordered: list[str] = []
    for cue in cues:
        if cue not in seen:
            seen.add(cue)
            ordered.append(cue)
    ctx.aggression_cues = ordered
    if ctx.sport_context:
        ctx.sport_confidence = kit_sport_confidence_boost(
            KitCues(team_kit_similarity=ctx.team_kit_similarity),
            ctx.sport_confidence,
        )
    return ctx


def should_soften_fight(ctx: SceneContext) -> bool:
    """True when sport / sports-venue context should suppress fight heuristics.

    Aimed-at-person firearm cues are never softened (unlike a bat on a field
    at low use-intensity).
    """
    if ctx.aimed_at_person or ctx.gunshot_proxy:
        return False
    if ctx.use_intensity_label == "possible_strike" and ctx.weapon_class:
        return False
    if ctx.thrown_at_person and ctx.throw_harmful and not ctx.throw_sport_projectile:
        return False
    if ctx.street_setting and ctx.sport_context and ctx.aggression_very_high:
        return False
    if ctx.sports_venue and ctx.sport_context and not ctx.aggression_very_high:
        return True
    return ctx.sport_decent and not ctx.aggression_high


def street_play_verify(ctx: SceneContext) -> bool:
    """Street + sport: still play unless aggression is extreme."""
    return bool(ctx.street_setting and ctx.sport_context and not ctx.aggression_very_high)


def intense_play_should_alert(override: Optional[bool] = None) -> bool:
    """Threat-page intense sport only when ALERT_ON_INTENSE_SPORT=1."""
    return alert_on_intense_sport(override)
