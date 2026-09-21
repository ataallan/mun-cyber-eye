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

    Soften when sport_context is at least decent **and** body-aggression
    is not high. A strong sports venue plus a named sport softens more
    readily. Street play still softens unless aggression is extreme.
    Strong aggression during sport is handled as intense play
    (stay game_or_play) rather than a heuristic fight alert — except
    street + extreme aggression, which leans fight for human review.
    """
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
