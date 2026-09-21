"""Risk classification for Mun Cyber Eye (Phase 2 heuristics + Phase 3 model).

Categories are provisional and require human verification. The system never
enforces access control, detention, or punishment autonomously.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

from vision.dataset import canonicalize_category
from vision.detector import Detection


class ActivityCategory(str, Enum):
    ORDINARY = "ordinary"
    GAME_OR_PLAY = "game_or_play"
    DANCE = "dance"
    POTENTIAL_FIGHT = "potential_fight"
    POTENTIAL_FALL = "potential_fall"
    POTENTIAL_WEAPON_OBJECT = "potential_weapon_object"


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

    def __init__(self, alert_on_game_or_dance: Optional[bool] = None) -> None:
        if alert_on_game_or_dance is None:
            alert_on_game_or_dance = _env_flag("ALERT_ON_GAME_OR_DANCE", "0")
        self.alert_on_game_or_dance = bool(alert_on_game_or_dance)

    FIGHT_SIGNALS = {
        "close_proximity",
        "rapid_motion",
        "strike_motion",
        "aggressive_pose",
    }
    FALL_SIGNALS = {"person_down", "horizontal_pose", "fall"}
    WEAPON_SIGNALS = {
        "knife",
        "gun",
        "firearm",
        "pistol",
        "rifle",
        "raised_object",
        "suspicious_object",
        "scissors",
        "baseball bat",
    }

    def assess(self, detections: Sequence[Detection]) -> RiskResult:
        labels = [d.label.lower() for d in detections]
        label_set = set(labels)
        conf_by_label = {}
        for d in detections:
            key = d.label.lower()
            conf_by_label[key] = max(conf_by_label.get(key, 0.0), d.confidence)

        # Phase 3: explicit activity-category detections (and aliases) take priority.
        activity_hits: list[tuple[Detection, ActivityCategory]] = []
        for d in detections:
            parsed = parse_activity_category(d.label)
            if parsed is not None:
                activity_hits.append((d, parsed))
        if activity_hits:
            top, category = max(activity_hits, key=lambda item: item[0].confidence)
            return self._from_activity(
                category,
                top.confidence,
                extras=top.extras,
            )

        fight_hits = label_set & self.FIGHT_SIGNALS
        fall_hits = label_set & self.FALL_SIGNALS
        weapon_hits = label_set & self.WEAPON_SIGNALS
        people = [d for d in detections if d.label.lower() == "person"]

        # Priority: weapon-object > fight > fall > ordinary
        if weapon_hits:
            conf = self._avg_conf(conf_by_label, weapon_hits, floor=0.55)
            level = RiskLevel.HIGH if conf >= 0.65 else RiskLevel.ELEVATED
            return RiskResult(
                category=ActivityCategory.POTENTIAL_WEAPON_OBJECT,
                risk_level=level,
                confidence=round(conf, 3),
                rationale=(
                    "Possible weapon-like or dangerous object indicators detected. "
                    "Requires human verification — not a determination of weapon possession."
                ),
                contributing_labels=sorted(weapon_hits | {p.label for p in people}),
                should_alert=True,
            )

        if len(fight_hits) >= 2 or (
            "strike_motion" in fight_hits and len(people) >= 2
        ):
            conf = self._avg_conf(conf_by_label, fight_hits, floor=0.6)
            # Boost slightly with multi-person context
            if len(people) >= 2:
                conf = min(0.95, conf + 0.08)
            level = RiskLevel.HIGH if conf >= 0.75 else RiskLevel.ELEVATED
            return RiskResult(
                category=ActivityCategory.POTENTIAL_FIGHT,
                risk_level=level,
                confidence=round(conf, 3),
                rationale=(
                    "Movement/interaction patterns consistent with a potential physical "
                    "confrontation. Alert for authorized human review only."
                ),
                contributing_labels=sorted(fight_hits | {p.label for p in people}),
                should_alert=True,
            )

        if fall_hits:
            conf = self._avg_conf(conf_by_label, fall_hits, floor=0.55)
            level = RiskLevel.ELEVATED if conf < 0.8 else RiskLevel.HIGH
            return RiskResult(
                category=ActivityCategory.POTENTIAL_FALL,
                risk_level=level,
                confidence=round(conf, 3),
                rationale=(
                    "Pose/orientation indicators consistent with a potential fall or "
                    "person down. Human verification required."
                ),
                contributing_labels=sorted(fall_hits | {p.label for p in people}),
                should_alert=True,
            )

        # Single fight signal → elevated, may still alert at lower priority
        if fight_hits and len(people) >= 2:
            conf = self._avg_conf(conf_by_label, fight_hits, floor=0.5)
            return RiskResult(
                category=ActivityCategory.POTENTIAL_FIGHT,
                risk_level=RiskLevel.ELEVATED,
                confidence=round(conf, 3),
                rationale=(
                    "Limited confrontation indicators with multiple people present. "
                    "Elevated attention; human review recommended."
                ),
                contributing_labels=sorted(fight_hits | {p.label for p in people}),
                should_alert=True,
            )

        person_conf = max((d.confidence for d in people), default=0.0)
        return RiskResult(
            category=ActivityCategory.ORDINARY,
            risk_level=RiskLevel.LOW,
            confidence=round(max(person_conf, 0.4) if people else 0.5, 3),
            rationale="No elevated risk indicators under current heuristics.",
            contributing_labels=sorted({d.label for d in detections}),
            should_alert=False,
        )

    def _from_activity(
        self,
        category: ActivityCategory,
        confidence: float,
        extras: dict | None = None,
    ) -> RiskResult:
        """Map a Phase 3 category prediction onto the Phase 2 risk contract."""
        conf = max(0.0, min(1.0, float(confidence)))
        extras = extras or {}
        scores = extras.get("scores") if isinstance(extras, dict) else None
        score_txt = ""
        if isinstance(scores, dict) and scores:
            ranked = sorted(scores.items(), key=lambda kv: -float(kv[1]))[:3]
            score_txt = " Model scores: " + ", ".join(
                f"{k}={float(v):.2f}" for k, v in ranked
            ) + "."

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
                rationale=self._non_threat_rationale(category, should_alert) + score_txt,
                contributing_labels=[category.value],
                should_alert=should_alert,
            )

        if category == ActivityCategory.POTENTIAL_WEAPON_OBJECT:
            level = RiskLevel.HIGH if conf >= 0.65 else RiskLevel.ELEVATED
            rationale = (
                "Phase 3 activity model flagged a potential weapon-like or dangerous "
                "object pattern. Requires human verification — not a determination "
                "of weapon possession."
                + score_txt
            )
        elif category == ActivityCategory.POTENTIAL_FIGHT:
            level = RiskLevel.HIGH if conf >= 0.75 else RiskLevel.ELEVATED
            rationale = (
                "Phase 3 activity model flagged movement/interaction patterns "
                "consistent with a potential physical confrontation (not game or "
                "play, and not dance). Alert for authorized human review only."
                + score_txt
            )
        else:
            level = RiskLevel.HIGH if conf >= 0.8 else RiskLevel.ELEVATED
            rationale = (
                "Phase 3 activity model flagged pose/orientation patterns "
                "consistent with a potential fall or person down. Human "
                "verification required."
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

    @staticmethod
    def _non_threat_rationale(category: ActivityCategory, should_alert: bool) -> str:
        queued = (
            "Alert queued because ALERT_ON_GAME_OR_DANCE is enabled. "
            if should_alert
            else "No threat alert queued. "
        )
        if category == ActivityCategory.GAME_OR_PLAY:
            return (
                "Phase 3 activity model classified the scene as game or play "
                "(sports, games, or playful roughhousing), not a fight. "
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

    @staticmethod
    def _avg_conf(conf_by_label: dict, keys: set, floor: float = 0.5) -> float:
        vals = [conf_by_label[k] for k in keys if k in conf_by_label]
        if not vals:
            return floor
        return max(floor, sum(vals) / len(vals))
