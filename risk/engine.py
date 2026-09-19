"""Risk classification heuristics for Mun Cyber Eye Phase 2.

Categories are provisional and require human verification. The system never
enforces access control, detention, or punishment autonomously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Sequence

from vision.detector import Detection


class ActivityCategory(str, Enum):
    ORDINARY = "ordinary"
    POTENTIAL_FIGHT = "potential_fight"
    POTENTIAL_FALL = "potential_fall"
    POTENTIAL_WEAPON_OBJECT = "potential_weapon_object"


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


class RiskEngine:
    """Heuristic risk engine combining detection labels into categories."""

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

    @staticmethod
    def _avg_conf(conf_by_label: dict, keys: set, floor: float = 0.5) -> float:
        vals = [conf_by_label[k] for k in keys if k in conf_by_label]
        if not vals:
            return floor
        return max(floor, sum(vals) / len(vals))
