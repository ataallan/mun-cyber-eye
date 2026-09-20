"""Responder-facing structured alert payload and advisory helpers.

Severity / recommended actions are advisory only. The system never enforces
access control, detention, or emergency dispatch on its own.
"""

from __future__ import annotations

import re
from typing import Any

SAFETY_BANNER = "AI detects and alerts. Humans verify and decide."
ENFORCEMENT_NOTICE = "none — advisory alert only; no autonomous enforcement"

# Responder-facing severity derived from Phase 2/3 risk_level.
SEVERITY_FROM_RISK = {
    "high": "critical",
    "elevated": "warning",
    "low": "info",
}

ADVISORY_ACTIONS = {
    "potential_fight": (
        "Advisory only: review the authorized feed now. If a physical "
        "confrontation is confirmed, follow site SOP and notify on-site "
        "personnel. This is not a determination of assault and is not a "
        "request to detain anyone."
    ),
    "potential_fall": (
        "Advisory only: review the feed for a possible person-down or medical "
        "event. If confirmed, dispatch trained human responders per SOP. The "
        "system will not call emergency services on its own."
    ),
    "potential_weapon_object": (
        "Advisory only: review the feed for a possible dangerous object. This "
        "is not proof of weapon possession or intent. If confirmed, follow "
        "site SOP and notify authorized personnel. Do not treat this as "
        "authorization to use force."
    ),
    "ordinary": (
        "Advisory only: no elevated response recommended. Continue normal "
        "monitoring."
    ),
}

DEFAULT_ADVISORY = (
    "Advisory only: a trained human must verify this alert before any "
    "consequential response. The system does not lock doors, dispatch force, "
    "or make legal determinations."
)

_DELIVERY_STATUSES = (
    "pending",
    "queued",
    "sent",
    "partial",
    "failed",
    "undelivered",
)


def severity_from_risk(risk_level: str) -> str:
    return SEVERITY_FROM_RISK.get((risk_level or "").lower(), "info")


def recommended_human_action(category: str) -> str:
    return ADVISORY_ACTIONS.get((category or "").lower(), DEFAULT_ADVISORY)


def short_rationale(rationale: str, limit: int = 160) -> str:
    text = (rationale or "").strip()
    if not text:
        return ""
    first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    if len(first) <= limit:
        return first
    return first[: limit - 1].rstrip() + "…"


def format_frame_time(timestamp_sec: float) -> str:
    try:
        total = max(0.0, float(timestamp_sec))
    except (TypeError, ValueError):
        return "00:00:00.00"
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def is_valid_delivery_status(status: str) -> bool:
    return status in _DELIVERY_STATUSES


def structured_payload(alert: Any) -> dict[str, Any]:
    """Canonical Phase 4 payload for UI, email, and SIEM/SOAR webhooks."""
    data = alert.to_dict() if hasattr(alert, "to_dict") else dict(alert)
    return {
        "id": data.get("id"),
        "correlation_id": data.get("correlation_id") or data.get("id"),
        "created_at": data.get("created_at"),
        "severity": data.get("severity") or severity_from_risk(data.get("risk_level", "")),
        "risk_level": data.get("risk_level"),
        "category": data.get("category"),
        "confidence": data.get("confidence"),
        "location_label": data.get("location_label") or "",
        "camera_id": data.get("camera_id") or "",
        "source": data.get("source_label") or data.get("source") or "",
        "frame_index": data.get("frame_index"),
        "frame_time": data.get("frame_time")
        or format_frame_time(data.get("timestamp_sec") or 0),
        "timestamp_sec": data.get("timestamp_sec"),
        "snapshot_path": data.get("snapshot_path"),
        "rationale": data.get("rationale"),
        "short_rationale": data.get("short_rationale")
        or short_rationale(data.get("rationale") or ""),
        "recommended_human_action": data.get("recommended_human_action")
        or recommended_human_action(data.get("category") or ""),
        "human_status": data.get("status"),
        "delivery_status": data.get("delivery_status") or "pending",
        "detections": data.get("detections") or [],
        "metadata": data.get("metadata") or {},
        "safety": SAFETY_BANNER,
        "enforcement": ENFORCEMENT_NOTICE,
    }
