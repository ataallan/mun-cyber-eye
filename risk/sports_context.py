"""Sport-versus-confrontation policy helpers.

When ``sport_context`` is present with decent confidence, fight heuristics
are softened unless body-aggression cues are strong. High aggression during
a named sport stays ``game_or_play`` (possible intense play) unless
``ALERT_ON_INTENSE_SPORT=1``.

This is assistive policy. It does not prove play is safe or that a scene
is a crime, and it never labels a face as criminal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from vision.detector import Detection
from vision.sports_catalog import resolve_sport, sport_display_name

SPORT_CONFIDENCE_DECENT = 0.55
SPORT_CONFIDENCE_STRONG = 0.70
AGGRESSION_HIGH = 0.65
AGGRESSION_VERY_HIGH = 0.85

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

    @property
    def sport_decent(self) -> bool:
        return bool(self.sport_context) and self.sport_confidence >= SPORT_CONFIDENCE_DECENT

    @property
    def sport_strong(self) -> bool:
        return bool(self.sport_context) and self.sport_confidence >= SPORT_CONFIDENCE_STRONG

    @property
    def aggression_high(self) -> bool:
        return self.aggression_score >= AGGRESSION_HIGH

    @property
    def aggression_very_high(self) -> bool:
        return self.aggression_score >= AGGRESSION_VERY_HIGH


def collect_scene_context(detections: Sequence[Detection]) -> SceneContext:
    """Read sport / aggression / face extras already attached to detections."""
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
    return ctx


def should_soften_fight(ctx: SceneContext) -> bool:
    """True when sport context should suppress Phase 2 fight heuristics.

    Soften when sport_context is at least decent **and** body-aggression
    is not high. Strong aggression during sport is handled as intense play
    (stay game_or_play) rather than a heuristic fight alert.
    """
    return ctx.sport_decent and not ctx.aggression_high


def intense_play_should_alert(override: Optional[bool] = None) -> bool:
    """Threat-page intense sport only when ALERT_ON_INTENSE_SPORT=1."""
    return alert_on_intense_sport(override)
