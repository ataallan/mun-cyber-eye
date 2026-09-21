"""Risk classification for Mun Cyber Eye (Phase 2 heuristics + Phase 3 model).

Categories are provisional and require human verification. The system never
enforces access control, detention, or punishment autonomously.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

from vision.dangerous_objects import detector_signal_labels, weapon_should_soften_sport
from vision.dataset import canonicalize_category
from vision.detector import Detection

from .sports_context import (
    SceneContext,
    collect_scene_context,
    should_soften_fight,
    street_play_verify,
)


class ActivityCategory(str, Enum):
    ORDINARY = "ordinary"
    GAME_OR_PLAY = "game_or_play"
    DANCE = "dance"
    POTENTIAL_FIGHT = "potential_fight"
    POTENTIAL_FALL = "potential_fall"
    POTENTIAL_WEAPON_OBJECT = "potential_weapon_object"
    POTENTIAL_GUNSHOT = "potential_gunshot"


# ordinary / game / dance are log-only unless ALERT_ON_GAME_OR_DANCE is enabled
_NON_THREAT_CATEGORIES = {
    ActivityCategory.ORDINARY,
    ActivityCategory.GAME_OR_PLAY,
    ActivityCategory.DANCE,
}


class RiskLevel(str, Enum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"


@dataclass
class RiskResult:
    category: ActivityCategory
    risk_level: RiskLevel
    confidence: float
    rationale: str
    contributing_labels: List[str] = field(default_factory=list)
    should_alert: bool = False
    sport_context: Optional[str] = None
    sport_confidence: float = 0.0
    sport_display: str = ""
    aggression_score: float = 0.0
    aggression_cues: List[str] = field(default_factory=list)
    face_cue_status: str = "disabled"
    face_note: str = ""
    place_type: str = "unknown"
    place_confidence: float = 0.0
    place_display: str = ""
    place_source: str = "none"
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
    thrown_at_person: bool = False
    throw_confidence: float = 0.0
    throw_label: str = ""


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def parse_activity_category(label: str) -> Optional[ActivityCategory]:
    """Map a detection label or alias onto a canonical activity category."""
    try:
        return ActivityCategory(canonicalize_category(label))
    except ValueError:
        return None


class RiskEngine:
    """Heuristic risk engine combining detection labels into categories."""

    def __init__(
        self,
        alert_on_game_or_dance: Optional[bool] = None,
        alert_on_intense_sport: Optional[bool] = None,
    ) -> None:
        if alert_on_game_or_dance is None:
            alert_on_game_or_dance = _env_flag("ALERT_ON_GAME_OR_DANCE", "0")
        if alert_on_intense_sport is None:
            alert_on_intense_sport = _env_flag("ALERT_ON_INTENSE_SPORT", "0")
        self.alert_on_game_or_dance = bool(alert_on_game_or_dance)
        self.alert_on_intense_sport = bool(alert_on_intense_sport)

    FIGHT_SIGNALS = {
        "close_proximity",
        "rapid_motion",
        "strike_motion",
        "aggressive_pose",
        "aggressive_motion",
    }
    FALL_SIGNALS = {"person_down", "horizontal_pose", "fall"}
    WEAPON_SIGNALS = detector_signal_labels()
    GUNSHOT_SIGNALS = {"possible_gunshot_video_proxy", "potential_gunshot"}

    def assess(self, detections: Sequence[Detection]) -> RiskResult:
        labels = [d.label.lower() for d in detections]
        label_set = set(labels)
        conf_by_label = {}
        for d in detections:
            key = d.label.lower()
            conf_by_label[key] = max(conf_by_label.get(key, 0.0), d.confidence)
        ctx = collect_scene_context(detections)

        # Aimed firearm-like object toward a person: never sport-softened.
        if ctx.aimed_at_person:
            return self._stamp(self._aimed_firearm_alert(ctx, detections), ctx)

        gunshot_hits = set(labels) & self.GUNSHOT_SIGNALS
        if ctx.gunshot_proxy or gunshot_hits:
            conf = max(
                ctx.gunshot_confidence,
                self._avg_conf(conf_by_label, gunshot_hits, floor=0.4) if gunshot_hits else 0.4,
            )
            return self._stamp(self._gunshot_alert(conf, ctx), ctx)

        if ctx.thrown_at_person and not self._throw_softens(ctx):
            return self._stamp(self._thrown_alert(ctx), ctx)

        # Phase 3: explicit activity-category detections (and aliases) take priority.
        activity_hits: list[tuple[Detection, ActivityCategory]] = []
        for d in detections:
            parsed = parse_activity_category(d.label)
            if parsed is not None:
                activity_hits.append((d, parsed))
        if activity_hits:
            top, category = max(activity_hits, key=lambda item: item[0].confidence)
            return self._stamp(
                self._from_activity(
                    category,
                    top.confidence,
                    extras=top.extras,
                    context=ctx,
                ),
                ctx,
            )

        fight_hits = label_set & self.FIGHT_SIGNALS
        fall_hits = label_set & self.FALL_SIGNALS
        weapon_hits = label_set & self.WEAPON_SIGNALS
        people = [d for d in detections if d.label.lower() == "person"]

        # Priority: weapon-object (unless sport-bat low intensity) > fight > fall
        if weapon_hits or self._weapon_present(ctx):
            if not self._weapon_sport_softens(ctx):
                conf = self._avg_conf(conf_by_label, weapon_hits, floor=0.55) if weapon_hits else max(
                    ctx.weapon_use_intensity, 0.55
                )
                if ctx.use_intensity_label == "possible_strike" or ctx.aimed_at_person:
                    conf = max(conf, 0.78)
                return self._stamp(
                    self._dangerous_object_alert(
                        ctx,
                        sorted(weapon_hits | {p.label for p in people} | {ctx.weapon_id or ""} - {""}),
                        confidence=conf,
                    ),
                    ctx,
                )

        fight_pattern = len(fight_hits) >= 2 or (
            "strike_motion" in fight_hits and len(people) >= 2
        )
        single_fight = bool(fight_hits) and len(people) >= 2
        if fight_pattern or single_fight:
            if street_play_verify(ctx) or (ctx.sport_decent and not (
                ctx.street_setting and ctx.aggression_very_high
            )):
                labels_out = sorted(
                    fight_hits
                    | set(ctx.aggression_cues)
                    | {p.label for p in people}
                    | {ctx.sport_context or "game_or_play"}
                )
                conf = max(
                    ctx.sport_confidence,
                    ctx.aggression_score,
                    self._avg_conf(conf_by_label, fight_hits, floor=0.55),
                )
                return self._stamp(
                    self._sport_vs_fight(confidence=conf, labels=labels_out, context=ctx),
                    ctx,
                )
            conf = self._avg_conf(conf_by_label, fight_hits, floor=0.6 if fight_pattern else 0.5)
            if fight_pattern and len(people) >= 2:
                conf = min(0.95, conf + 0.08)
            level = (
                RiskLevel.HIGH
                if (fight_pattern and conf >= 0.75) or ctx.strong_confrontation_setting
                else RiskLevel.ELEVATED
            )
            setting_bit = self._setting_rationale(ctx, lean_fight=True)
            rationale = (
                "Movement/interaction patterns consistent with a potential physical "
                "confrontation (no named sport context). Alert for authorized human "
                "review only."
                + setting_bit
                if fight_pattern
                else (
                    "Limited confrontation indicators with multiple people present "
                    "and no named sport context. Elevated attention; human review "
                    "recommended."
                    + setting_bit
                )
            )
            return self._stamp(
                RiskResult(
                    category=ActivityCategory.POTENTIAL_FIGHT,
                    risk_level=level,
                    confidence=round(conf, 3),
                    rationale=rationale,
                    contributing_labels=sorted(fight_hits | {p.label for p in people}),
                    should_alert=True,
                ),
                ctx,
            )

        if fall_hits:
            conf = self._avg_conf(conf_by_label, fall_hits, floor=0.55)
            level = RiskLevel.ELEVATED if conf < 0.8 else RiskLevel.HIGH
            return self._stamp(
                RiskResult(
                    category=ActivityCategory.POTENTIAL_FALL,
                    risk_level=level,
                    confidence=round(conf, 3),
                    rationale=(
                        "Pose/orientation indicators consistent with a potential fall or "
                        "person down. Human verification required."
                        + self._fall_manner_suffix(ctx)
                    ),
                    contributing_labels=sorted(fall_hits | {p.label for p in people}),
                    should_alert=True,
                ),
                ctx,
            )

        if ctx.aggression_high and not ctx.sport_decent and not street_play_verify(ctx):
            return self._stamp(
                self._aggression_fight(
                    people_labels=[p.label for p in people],
                    ctx=ctx,
                    extra="",
                ),
                ctx,
            )

        if ctx.sport_decent:
            return self._stamp(
                self._sport_vs_fight(
                    confidence=max(ctx.sport_confidence, 0.5),
                    labels=sorted({ctx.sport_context or "game_or_play"} | {p.label for p in people}),
                    context=ctx,
                ),
                ctx,
            )

        person_conf = max((d.confidence for d in people), default=0.0)
        return self._stamp(
            RiskResult(
                category=ActivityCategory.ORDINARY,
                risk_level=RiskLevel.LOW,
                confidence=round(max(person_conf, 0.4) if people else 0.5, 3),
                rationale="No elevated risk indicators under current heuristics.",
                contributing_labels=sorted({d.label for d in detections}),
                should_alert=False,
            ),
            ctx,
        )

    def _from_activity(
        self,
        category: ActivityCategory,
        confidence: float,
        extras: dict | None = None,
        context: Optional[SceneContext] = None,
    ) -> RiskResult:
        """Map a Phase 3 category prediction onto the Phase 2 risk contract."""
        conf = max(0.0, min(1.0, float(confidence)))
        extras = extras or {}
        ctx = context or SceneContext()
        scores = extras.get("scores") if isinstance(extras, dict) else None
        score_txt = ""
        if isinstance(scores, dict) and scores:
            ranked = sorted(scores.items(), key=lambda kv: -float(kv[1]))[:3]
            score_txt = " Model scores: " + ", ".join(
                f"{k}={float(v):.2f}" for k, v in ranked
            ) + "."

        if category == ActivityCategory.GAME_OR_PLAY:
            if ctx.aimed_at_person:
                return self._aimed_firearm_alert(ctx, [])
            if ctx.thrown_at_person and not self._throw_softens(ctx):
                return self._thrown_alert(ctx)
            if self._weapon_present(ctx) and not self._weapon_sport_softens(ctx):
                return self._dangerous_object_alert(ctx, [category.value])
            return self._combine_game_and_aggression(conf, score_txt, ctx)

        if (
            category == ActivityCategory.ORDINARY
            and ctx.aggression_high
            and not ctx.sport_decent
            and not street_play_verify(ctx)
        ):
            return self._aggression_fight(
                people_labels=["ordinary"],
                ctx=ctx,
                extra=(
                    " The activity model had said ordinary."
                    + score_txt
                ),
            )

        if category == ActivityCategory.POTENTIAL_FIGHT and ctx.sport_strong and not ctx.aggression_high:
            return self._sport_vs_fight(
                confidence=max(conf, ctx.sport_confidence),
                labels=["potential_fight", ctx.sport_context or "game_or_play"],
                context=ctx,
                score_txt=score_txt,
                softened_from_fight=True,
            )

        if category in _NON_THREAT_CATEGORIES:
            should_alert = False
            level = RiskLevel.LOW
            if category != ActivityCategory.ORDINARY and self.alert_on_game_or_dance:
                should_alert = True
                level = RiskLevel.ELEVATED
            return RiskResult(
                category=category,
                risk_level=level,
                confidence=round(max(conf, 0.4), 3),
                rationale=self._non_threat_rationale(category, should_alert, ctx) + score_txt,
                contributing_labels=[category.value],
                should_alert=should_alert,
            )

        if category == ActivityCategory.POTENTIAL_WEAPON_OBJECT:
            if self._weapon_sport_softens(ctx):
                return self._sport_vs_fight(
                    confidence=max(conf, ctx.sport_confidence, 0.5),
                    labels=[category.value, ctx.sport_context or "game_or_play"],
                    context=ctx,
                    score_txt=score_txt,
                )
            level = RiskLevel.HIGH if conf >= 0.65 or ctx.use_intensity_label == "possible_strike" else RiskLevel.ELEVATED
            rationale = (
                "Phase 3 activity model flagged a potential weapon-like or dangerous "
                "object pattern. Requires human verification — not a determination "
                "of weapon possession."
                + self._weapon_suffix(ctx)
                + score_txt
            )
        elif category == ActivityCategory.POTENTIAL_FIGHT:
            level = RiskLevel.HIGH if conf >= 0.75 else RiskLevel.ELEVATED
            rationale = (
                "Phase 3 activity model flagged movement/interaction patterns "
                "consistent with a potential physical confrontation (not game or "
                "play, and not dance). Alert for authorized human review only."
                + self._face_rationale_suffix(ctx)
                + score_txt
            )
        else:
            level = RiskLevel.HIGH if conf >= 0.8 else RiskLevel.ELEVATED
            rationale = (
                "Phase 3 activity model flagged pose/orientation patterns "
                "consistent with a potential fall or person down. Human "
                "verification required."
                + self._fall_manner_suffix(ctx)
                + score_txt
            )

        return RiskResult(
            category=category,
            risk_level=level,
            confidence=round(conf, 3),
            rationale=rationale,
            contributing_labels=[category.value],
            should_alert=True,
        )

    def _combine_game_and_aggression(
        self,
        confidence: float,
        score_txt: str,
        ctx: SceneContext,
    ) -> RiskResult:
        if ctx.aggression_high and not ctx.sport_decent and not street_play_verify(ctx):
            return self._aggression_fight(
                people_labels=["game_or_play"],
                ctx=ctx,
                extra=(
                    " Leaning potential confrontation rather than game or play."
                    + self._face_rationale_suffix(ctx)
                    + score_txt
                ),
            )
        if ctx.aggression_high and ctx.sport_decent:
            return self._sport_vs_fight(
                confidence=max(confidence, ctx.sport_confidence, ctx.aggression_score),
                labels=["game_or_play", ctx.sport_context or "game_or_play", *ctx.aggression_cues],
                context=ctx,
                score_txt=score_txt,
            )
        should_alert = bool(self.alert_on_game_or_dance)
        return RiskResult(
            category=ActivityCategory.GAME_OR_PLAY,
            risk_level=RiskLevel.ELEVATED if should_alert else RiskLevel.LOW,
            confidence=round(max(confidence, 0.4), 3),
            rationale=self._non_threat_rationale(
                ActivityCategory.GAME_OR_PLAY, should_alert, ctx
            )
            + score_txt,
            contributing_labels=["game_or_play"]
            + ([ctx.sport_context] if ctx.sport_context else []),
            should_alert=should_alert,
        )

    def _sport_vs_fight(
        self,
        *,
        confidence: float,
        labels: list[str],
        context: SceneContext,
        score_txt: str = "",
        softened_from_fight: bool = False,
    ) -> RiskResult:
        """Named sport present: stay game_or_play unless operators page intense play."""
        sport_name = context.sport_display or context.sport_context or "a catalog sport"
        if context.street_setting and context.aggression_very_high:
            return RiskResult(
                category=ActivityCategory.POTENTIAL_FIGHT,
                risk_level=RiskLevel.HIGH,
                confidence=round(max(confidence, context.aggression_score, 0.7), 3),
                rationale=(
                    f"Street setting plus extreme body-aggression "
                    f"({context.aggression_score:.2f}) despite sport context "
                    f"{sport_name}. Leaning potential confrontation for authorized "
                    "human review — not a determination of assault."
                    + self._setting_rationale(context, lean_fight=True)
                    + self._kit_rationale_suffix(context)
                    + self._face_rationale_suffix(context)
                    + score_txt
                ),
                contributing_labels=labels,
                should_alert=True,
            )
        if context.aggression_high:
            should_alert = bool(self.alert_on_intense_sport)
            queued = (
                "Info-level alert queued because ALERT_ON_INTENSE_SPORT is enabled. "
                if should_alert
                else "No threat alert queued. "
            )
            lead = (
                "Activity model said confrontation, but a named sport context is "
                "present and body-aggression is not decisive — treating as possible "
                "intense play. "
                if softened_from_fight
                else "Possible intense play; human should verify. "
            )
            if street_play_verify(context):
                lead = "Street play — verify. " + lead
            elif context.sports_venue:
                lead = (
                    "Sports venue plus named sport — stronger lean toward game or play. "
                    + lead
                )
            return RiskResult(
                category=ActivityCategory.GAME_OR_PLAY,
                risk_level=RiskLevel.ELEVATED if should_alert else RiskLevel.LOW,
                confidence=round(max(confidence, 0.5), 3),
                rationale=(
                    f"{lead}Sport context: {sport_name} "
                    f"(confidence {context.sport_confidence:.2f}). "
                    f"Body-aggression score {context.aggression_score:.2f} "
                    f"({', '.join(context.aggression_cues) or 'motion proxies'}). "
                    + queued
                    + "Play can look like a clash; this is not a determination of assault."
                    + self._setting_rationale(context, lean_fight=False)
                    + self._kit_rationale_suffix(context)
                    + self._face_rationale_suffix(context)
                    + score_txt
                ),
                contributing_labels=labels,
                should_alert=should_alert,
            )
        should_alert = bool(self.alert_on_game_or_dance)
        lead = (
            "Confrontation cues were softened because a named sport context is "
            "present and body-aggression is not high. "
            if (softened_from_fight or should_soften_fight(context))
            else ""
        )
        if street_play_verify(context):
            lead = "Street play — verify. " + lead
        elif context.sports_venue and context.sport_context:
            lead = (
                "Sports venue plus named sport — stronger lean toward game or play. "
                + lead
            )
        return RiskResult(
            category=ActivityCategory.GAME_OR_PLAY,
            risk_level=RiskLevel.ELEVATED if should_alert else RiskLevel.LOW,
            confidence=round(max(confidence, 0.45), 3),
            rationale=(
                f"{lead}Scene matches game or play"
                f" ({sport_name}). No threat alert queued. "
                "Human operators may still review the live source."
                + self._setting_rationale(context, lean_fight=False)
                + self._kit_rationale_suffix(context)
                + self._face_rationale_suffix(context)
                + score_txt
            ),
            contributing_labels=labels,
            should_alert=should_alert,
        )

    def _aggression_fight(
        self,
        *,
        people_labels: list[str],
        ctx: SceneContext,
        extra: str = "",
    ) -> RiskResult:
        level = (
            RiskLevel.HIGH
            if ctx.strong_confrontation_setting
            else RiskLevel.ELEVATED
        )
        setting = self._setting_rationale(ctx, lean_fight=True)
        return RiskResult(
            category=ActivityCategory.POTENTIAL_FIGHT,
            risk_level=level,
            confidence=round(max(ctx.aggression_score, 0.6), 3),
            rationale=(
                "Sustained high motion / body-aggression proxies without a named "
                "sport context. Leaning potential confrontation for authorized "
                "human review — not a determination of assault."
                + setting
                + extra
                + self._kit_rationale_suffix(ctx)
                + self._face_rationale_suffix(ctx)
            ),
            contributing_labels=sorted(set(people_labels) | set(ctx.aggression_cues)),
            should_alert=True,
        )

    @staticmethod
    def _setting_rationale(ctx: SceneContext, *, lean_fight: bool) -> str:
        if not ctx.place_type or ctx.place_type == "unknown":
            return ""
        label = ctx.place_display or ctx.place_type
        source = f" ({ctx.place_source})" if ctx.place_source not in {"", "none"} else ""
        if lean_fight and ctx.strong_confrontation_setting and not ctx.sport_decent:
            return (
                f" Setting: {label}{source} — street / corridor / house / compound "
                "plus high body-aggression and no sport context, so a stronger lean "
                "toward potential_fight. Place is a catalog type, not a named venue."
            )
        if street_play_verify(ctx):
            return (
                f" Setting: {label}{source}. Street play is still treated as "
                "game or play unless aggression is extreme — verify."
            )
        return (
            f" Setting: {label}{source}. Catalog place type only — never a "
            "specific arena name."
        )

    @staticmethod
    def _kit_rationale_suffix(ctx: SceneContext) -> str:
        if ctx.kit_supports_play:
            return (
                f" Similar clothing colors (kit similarity {ctx.team_kit_similarity:.2f}) "
                "slightly support a team / play reading — not identity or guilt."
            )
        if ctx.jersey_like_colors:
            return (
                " Saturated jersey-like colors noted; absence of matching kits "
                "does not prove a fight."
            )
        return ""

    @staticmethod
    def _face_rationale_suffix(ctx: SceneContext) -> str:
        if ctx.face_status == "assistive" and ctx.face_note:
            return " " + ctx.face_note
        return ""

    @staticmethod
    def _non_threat_rationale(
        category: ActivityCategory,
        should_alert: bool,
        ctx: Optional[SceneContext] = None,
    ) -> str:
        queued = (
            "Alert queued because ALERT_ON_GAME_OR_DANCE is enabled. "
            if should_alert
            else "No threat alert queued. "
        )
        sport_bit = ""
        if ctx and ctx.sport_context:
            sport_bit = (
                f" Sport context: {ctx.sport_display or ctx.sport_context}"
                f" ({ctx.sport_confidence:.2f})."
            )
        if ctx and ctx.place_type and ctx.place_type != "unknown":
            sport_bit += (
                f" Setting: {ctx.place_display or ctx.place_type}"
                f" ({ctx.place_source or 'assist'})."
            )
        if ctx and street_play_verify(ctx):
            sport_bit += " Street play — verify."
        if category == ActivityCategory.GAME_OR_PLAY:
            return (
                "Phase 3 activity model classified the scene as game or play "
                "(sports, games, or playful roughhousing), not a fight."
                + sport_bit
                + " "
                + queued
                + "Human operators may still review the live source."
            )
        if category == ActivityCategory.DANCE:
            return (
                "Phase 3 activity model classified the scene as dance / "
                "choreographed movement, not a confrontation. "
                + queued
                + "Human operators may still review the live source."
            )
        return (
            "Phase 3 activity model classified the scene as ordinary. "
            "No alert queued. Human operators may still review the live source."
        )

    def _weapon_present(self, ctx: SceneContext) -> bool:
        if ctx.aimed_at_person:
            return True
        if ctx.weapon_id and ctx.use_intensity_label and ctx.use_intensity_label != "none":
            return True
        return bool(ctx.weapon_class and ctx.weapon_use_intensity > 0)

    def _weapon_sport_softens(self, ctx: SceneContext) -> bool:
        return weapon_should_soften_sport(
            weapon_id=ctx.weapon_id or "",
            sport_context=ctx.sport_context,
            sports_venue=ctx.sports_venue,
            sport_decent=ctx.sport_decent,
            use_intensity=ctx.weapon_use_intensity,
            use_intensity_label=ctx.use_intensity_label or "",
            aimed_at_person=ctx.aimed_at_person,
            gunshot_proxy=ctx.gunshot_proxy,
        )

    def _dangerous_object_alert(
        self,
        ctx: SceneContext,
        labels: list[str],
        confidence: Optional[float] = None,
    ) -> RiskResult:
        conf = max(
            0.55,
            float(confidence or 0),
            ctx.weapon_use_intensity,
        )
        high = (
            ctx.harm_potential == "high"
            or ctx.use_intensity_label in {"possible_strike", "threatening_motion"}
            or ctx.aimed_at_person
            or conf >= 0.65
        )
        what = ctx.weapon_id or "dangerous object"
        klass = ctx.weapon_class or "unknown"
        harm = ctx.harm_potential or "unknown"
        intensity_label = ctx.use_intensity_label or ctx.weapon_use_tier or "none"
        return RiskResult(
            category=ActivityCategory.POTENTIAL_WEAPON_OBJECT,
            risk_level=RiskLevel.HIGH if high else RiskLevel.ELEVATED,
            confidence=round(min(0.95, conf), 3),
            rationale=(
                f"Possible weapon-like or dangerous object ({what}, class {klass}, "
                f"harm_potential {harm}). Use-against-person intensity "
                f"{ctx.weapon_use_intensity:.2f} ({intensity_label}) — indicator only, "
                "not proof of assault or intent. Toys, tools, and phones false-fire. "
                "Requires human verification — not a determination of weapon possession."
                + self._weapon_suffix(ctx)
            ),
            contributing_labels=sorted({lab for lab in labels if lab}),
            should_alert=True,
        )

    def _throw_softens(self, ctx: SceneContext) -> bool:
        from vision.throw_assist import ThrowAssessment, throw_should_soften

        assessment = ThrowAssessment(
            thrown_at_person=ctx.thrown_at_person,
            confidence=ctx.throw_confidence,
            harmful=ctx.throw_harmful,
            sport_projectile=ctx.throw_sport_projectile,
        )
        return throw_should_soften(
            assessment,
            sport_context=ctx.sport_context,
            sports_venue=ctx.sports_venue,
            aimed_at_person=ctx.aimed_at_person,
            aggression_high=ctx.aggression_high,
            confrontation_setting=ctx.confrontation_setting
            or ctx.strong_confrontation_setting,
        )

    def _aimed_firearm_alert(
        self, ctx: SceneContext, detections: Sequence[Detection]
    ) -> RiskResult:
        labels = sorted({d.label.lower() for d in detections})
        return RiskResult(
            category=ActivityCategory.POTENTIAL_WEAPON_OBJECT,
            risk_level=RiskLevel.HIGH,
            confidence=round(max(0.78, ctx.weapon_use_intensity, 0.7), 3),
            rationale=(
                "Possible firearm-like object pointed toward a person — verify. "
                "Not proof of a real firearm or intent. "
                "Sport context does not suppress this cue. Humans must review. "
                "The system does not enforce or dispatch."
            ),
            contributing_labels=labels,
            should_alert=True,
        )

    def _gunshot_alert(self, confidence: float, ctx: SceneContext) -> RiskResult:
        conf = max(0.36, min(0.85, float(confidence) or ctx.gunshot_confidence or 0.4))
        level = RiskLevel.HIGH if conf >= 0.55 else RiskLevel.ELEVATED
        return RiskResult(
            category=ActivityCategory.POTENTIAL_GUNSHOT,
            risk_level=level,
            confidence=round(conf, 3),
            rationale=(
                "Possible gunshot video proxy (localized flash / dive / firearm-like "
                "object) — verify, not a confirmed gunshot. Not forensic ballistic "
                "proof. Fireworks, reflections, and camera artifacts false-fire. "
                f"Audio status: {ctx.gunshot_audio_status or 'disabled'}. "
                "Humans review. The system does not dispatch or enforce."
            ),
            contributing_labels=["possible_gunshot_video_proxy"],
            should_alert=True,
        )

    def _thrown_alert(self, ctx: SceneContext) -> RiskResult:
        conf = max(0.45, float(ctx.throw_confidence) or 0.5)
        level = RiskLevel.HIGH if ctx.throw_harmful or conf >= 0.65 else RiskLevel.ELEVATED
        what = ctx.throw_label or "object"
        return RiskResult(
            category=ActivityCategory.POTENTIAL_WEAPON_OBJECT,
            risk_level=level,
            confidence=round(conf, 3),
            rationale=(
                f"Possible object thrown toward a person ({what}) — verify. "
                "Not proof of assault. Distinct from an aimed firearm and from a "
                "static brandish. Humans must review. No enforcement."
            ),
            contributing_labels=sorted(
                {"object_thrown_at_person", what} | set(ctx.throw_cues)
            ),
            should_alert=True,
        )

    @staticmethod
    def _fall_manner_suffix(ctx: SceneContext) -> str:
        if not ctx.fall_manner:
            return (
                " Fall manner unknown (weak cues). Vision cannot medically "
                "diagnose syncope vs assault vs trip."
            )
        display = ctx.fall_display or ctx.fall_manner.replace("_", " ")
        return (
            f" Fall manner assist: {display} "
            f"(confidence {ctx.fall_confidence:.2f}). Not a medical diagnosis "
            "of syncope, assault, or a trip."
        )

    @staticmethod
    def _weapon_suffix(ctx: SceneContext) -> str:
        bits = []
        if ctx.aimed_at_person:
            bits.append(" Aimed-at-person cue is present.")
        if ctx.thrown_at_person:
            bits.append(" Thrown-object-toward-person cue is present.")
        if ctx.weapon_use_tier or ctx.use_intensity_label:
            bits.append(
                f" Use intensity {ctx.weapon_use_intensity:.2f} "
                f"({ctx.use_intensity_label or ctx.weapon_use_tier})."
            )
        if ctx.weapon_class:
            bits.append(
                f" Weapon class {ctx.weapon_class}, harm_potential "
                f"{ctx.harm_potential or 'unknown'}."
            )
        return "".join(bits)

    @staticmethod
    def _stamp(result: RiskResult, ctx: SceneContext) -> RiskResult:
        result.sport_context = ctx.sport_context
        result.sport_confidence = ctx.sport_confidence
        result.sport_display = ctx.sport_display
        result.aggression_score = ctx.aggression_score
        result.aggression_cues = list(ctx.aggression_cues)
        result.face_cue_status = ctx.face_status
        result.face_note = ctx.face_note
        result.place_type = ctx.place_type or "unknown"
        result.place_confidence = ctx.place_confidence
        result.place_display = ctx.place_display or ""
        result.place_source = ctx.place_source or "none"
        result.team_kit_similarity = ctx.team_kit_similarity
        result.jersey_like_colors = ctx.jersey_like_colors
        result.kit_note = ctx.kit_note
        result.fall_manner = ctx.fall_manner
        result.fall_confidence = ctx.fall_confidence
        result.fall_display = ctx.fall_display
        result.gunshot_proxy = ctx.gunshot_proxy
        result.gunshot_confidence = ctx.gunshot_confidence
        result.gunshot_audio_status = ctx.gunshot_audio_status
        result.aimed_at_person = ctx.aimed_at_person
        result.weapon_use_intensity = ctx.weapon_use_intensity
        result.weapon_use_tier = ctx.weapon_use_tier
        result.use_intensity_label = ctx.use_intensity_label
        result.weapon_class = ctx.weapon_class
        result.harm_potential = ctx.harm_potential
        result.weapon_id = ctx.weapon_id
        result.thrown_at_person = ctx.thrown_at_person
        result.throw_confidence = ctx.throw_confidence
        result.throw_label = ctx.throw_label
        return result

    @staticmethod
    def _avg_conf(conf_by_label: dict, keys: set, floor: float = 0.5) -> float:
        vals = [conf_by_label[k] for k in keys if k in conf_by_label]
        if not vals:
            return floor
        return max(floor, sum(vals) / len(vals))
