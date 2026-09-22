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

CATEGORY_DISPLAY_NAMES = {
    "ordinary": "ordinary",
    "game_or_play": "game or play",
    "dance": "dance",
    "potential_fight": "potential confrontation",
    "potential_fall": "potential fall",
    "potential_weapon_object": "potential weapon-like object",
    "potential_gunshot": "potential gunshot (video proxy)",
}

ADVISORY_ACTIONS = {
    "potential_fight": (
        "Advisory only: review the feed. If a confrontation is confirmed, "
        "follow site procedure."
    ),
    "potential_fall": (
        "Advisory only: review the feed for a person down. If confirmed, "
        "follow site procedure."
    ),
    "potential_weapon_object": (
        "Advisory only: review the feed for a dangerous object. "
        "Not proof of intent. If confirmed, follow site procedure."
    ),
    "potential_gunshot": (
        "Advisory only: review the feed for a possible gunshot. "
        "This is not a confirmed gunshot. If confirmed, follow site procedure."
    ),
    "ordinary": "Advisory only: no elevated response recommended.",
    "game_or_play": (
        "Advisory only: classified as game or play, not a fight. "
        "No threat response recommended."
    ),
    "dance": (
        "Advisory only: classified as dance, not a confrontation. "
        "No threat response recommended."
    ),
}

DEFAULT_ADVISORY = "Advisory only: verify this alert before any response."

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


def category_display_name(category: str) -> str:
    """Plain-language label for console / email (slug stays on the payload)."""
    key = (category or "").strip().lower()
    if key in CATEGORY_DISPLAY_NAMES:
        return CATEGORY_DISPLAY_NAMES[key]
    return (category or "").replace("_", " ").strip() or "unknown"


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
        "category_label": category_display_name(data.get("category") or ""),
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
