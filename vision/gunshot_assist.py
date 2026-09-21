"""Gunshot *video proxy* assist — honest about the acoustic gap.

Reliable gunshot detection is usually **acoustic**. This prototype is
video-first. We only emit a low-moderate ``possible_gunshot_video_proxy``
from visual cues (localized flash, multi-person dive, firearm-like object).
Optional audio is off by default and reports ``unavailable`` rather than
inventing a bang.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from vision.aggression import AggressionAssessment
from vision.detector import Detection

ENV_ENABLE_GUNSHOT_AUDIO = "ENABLE_GUNSHOT_AUDIO"

FIREARM_LABELS = frozenset(
    {
        "gun",
        "firearm",
        "pistol",
        "rifle",
        "handgun",
        "firearm_aimed_at_person",
    }
)
GUNSHOT_PROXY_LABEL = "possible_gunshot_video_proxy"

VIDEO_NOTE = (
    "Possible gunshot *video proxy* only — not a confirmed gunshot, not "
    "forensic ballistic proof. Flashes, fireworks, and reflections false-fire. "
    "Humans verify. The system does not dispatch or enforce."
)
AUDIO_DISABLED_NOTE = (
    "Gunshot audio assist is off (ENABLE_GUNSHOT_AUDIO=0). "
    "No audio detections were invented."
)
AUDIO_UNAVAILABLE_NOTE = (
    "Gunshot audio side-channel unavailable (no wav / microphone). "
    "No audio detections were invented."
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def gunshot_audio_enabled() -> bool:
    return _env_flag(ENV_ENABLE_GUNSHOT_AUDIO, "0")


@dataclass
class GunshotAssessment:
    video_proxy: bool = False
    confidence: float = 0.0
    cues: list[str] = field(default_factory=list)
    event_type: str = ""
    note: str = (
        "No gunshot video proxy on this frame. Audio is separate and off "
        "by default."
    )
    audio_status: str = "disabled"
    audio_confidence: float = 0.0
    audio_note: str = AUDIO_DISABLED_NOTE

    def to_dict(self) -> dict:
        return {
            "video_proxy": bool(self.video_proxy),
            "confidence": round(float(self.confidence), 3),
            "cues": list(self.cues),
            "event_type": self.event_type,
            "note": self.note,
            "audio_status": self.audio_status,
            "audio_confidence": round(float(self.audio_confidence), 3),
            "audio_note": self.audio_note,
        }


def _as_small(image_bgr: np.ndarray, size: tuple[int, int] = (80, 60)) -> np.ndarray:
    import cv2

    if image_bgr.ndim == 2:
        bgr = image_bgr
    else:
        bgr = image_bgr
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)


def _localized_flash(
    image_bgr: np.ndarray,
    prev_bgr: Optional[np.ndarray],
) -> tuple[bool, float]:
    """True when a small region spikes in brightness vs a stable median."""
    import cv2

    if (
        prev_bgr is None
        or getattr(prev_bgr, "size", 0) == 0
        or image_bgr is None
        or getattr(image_bgr, "size", 0) == 0
    ):
        return False, 0.0
    cur = _as_small(image_bgr)
    prev = _as_small(prev_bgr)
    if cur.shape != prev.shape:
        prev = cv2.resize(prev, (cur.shape[1], cur.shape[0]))
    cur_v = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    prev_v = cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    delta = cur_v - prev_v
    h, w = delta.shape
    cells = []
    for y in range(8):
        for x in range(8):
            block = delta[y * h // 8 : (y + 1) * h // 8, x * w // 8 : (x + 1) * w // 8]
            cells.append(float(block.mean()))
    arr = np.array(cells, dtype=np.float32)
    peak = float(arr.max()) if arr.size else 0.0
    med = float(np.median(arr)) if arr.size else 0.0
    # Whole-frame lighting shift (MOCK color-wash, fireworks sky) is uniform.
    if peak < 28.0:
        return False, 0.0
    if peak < med * 2.4 + 12.0:
        return False, 0.0
    score = float(min(0.55, (peak - 28.0) / 90.0))
    return True, score


def analyze_gunshot_audio(wav_path: str | Path | None = None) -> dict:
    """Optional audio stub. Never invents a gunshot when disabled or missing."""
    if not gunshot_audio_enabled():
        return {
            "status": "disabled",
            "confidence": 0.0,
            "note": AUDIO_DISABLED_NOTE,
        }
    path = Path(wav_path) if wav_path else None
    if path is None or not path.is_file():
        return {
            "status": "unavailable",
            "confidence": 0.0,
            "note": AUDIO_UNAVAILABLE_NOTE,
        }
    # No production acoustic classifier is bundled. A present wav without a
    # detector is still unavailable — do not treat file existence as a bang.
    return {
        "status": "unavailable",
        "confidence": 0.0,
        "note": (
            "Audio file present but no gunshot acoustic model is bundled. "
            "No audio detections were invented."
        ),
    }


def analyze_gunshot_proxy(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
    aggression: Optional[AggressionAssessment] = None,
    prev_bgr: Optional[np.ndarray] = None,
    *,
    audio_wav: str | Path | None = None,
) -> GunshotAssessment:
    """Low-moderate visual proxy. Audio is a separate gated stub."""
    audio = analyze_gunshot_audio(audio_wav)
    result = GunshotAssessment(
        audio_status=str(audio.get("status") or "disabled"),
        audio_confidence=float(audio.get("confidence") or 0.0),
        audio_note=str(audio.get("note") or AUDIO_DISABLED_NOTE),
    )
    aggression = aggression or AggressionAssessment()
    labels = {str(d.label).lower() for d in detections}
    people = [d for d in detections if str(d.label).lower() == "person"]
    firearm = bool(labels & FIREARM_LABELS)
    flash, flash_score = _localized_flash(image_bgr, prev_bgr)
    dive = (
        len(people) >= 2
        and (
            float(aggression.sudden_acceleration) >= 0.40
            or float(aggression.motion_intensity) >= 0.38
        )
        and not aggression.uniform_scene_change
    )
    cues: list[str] = []
    conf = 0.0
    if flash:
        cues.append("localized_flash")
        conf += 0.28 + flash_score * 0.2
    if dive:
        cues.append("multi_person_dive")
        conf += 0.22
    if firearm:
        cues.append("firearm_like_object")
        conf += 0.18
    # Require a combination — a lone color-wash or a lone knife is not a gunshot.
    combo = (flash and dive) or (flash and firearm) or (firearm and dive and flash)
    if not combo:
        result.cues = cues
        result.note = (
            "No combined gunshot video proxy (need localized flash plus "
            "dive and/or firearm-like object). "
            + VIDEO_NOTE
        )
        return result
    conf = float(min(0.62, max(0.36, conf)))
    result.video_proxy = True
    result.confidence = round(conf, 3)
    result.cues = cues
    result.event_type = "possible_gunshot"
    result.note = VIDEO_NOTE
    return result


def detections_from_gunshot(
    assessment: GunshotAssessment,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list[Detection]:
    if not assessment.video_proxy:
        return []
    return [
        Detection(
            label=GUNSHOT_PROXY_LABEL,
            confidence=float(assessment.confidence),
            bbox_xyxy=bbox,
            extras={
                "event_type": "possible_gunshot",
                "gunshot": assessment.to_dict(),
            },
        )
    ]
