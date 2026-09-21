"""Optional, gated face-expression assist — off by default.

Never identifies a person, never infers demographics, and never attaches
a criminal label. When disabled the pipeline must not analyze faces.
A robust FER model is not bundled; if Haar cascades are missing we report
``unavailable`` instead of inventing scores.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ENV_ENABLE_FACE_AGGRESSION = "ENABLE_FACE_AGGRESSION"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def face_aggression_enabled(override: Optional[bool] = None) -> bool:
    if override is not None:
        return bool(override)
    return _env_flag(ENV_ENABLE_FACE_AGGRESSION, "0")


@dataclass
class FaceAggressionResult:
    status: str = "disabled"  # disabled | none_detected | unavailable | assistive
    score: Optional[float] = None
    note: str = ""
    faces_found: int = 0

    def to_dict(self) -> dict:
        return {
            "face_aggression": self.status
            if self.status in {"disabled", "unavailable", "none_detected"}
            else "assistive",
            "status": self.status,
            "score": None if self.score is None else round(float(self.score), 3),
            "note": self.note,
            "faces_found": int(self.faces_found),
            "identity": None,
            "demographics": None,
            "criminal_label": None,
        }


def _disabled_result() -> FaceAggressionResult:
    return FaceAggressionResult(
        status="disabled",
        score=None,
        note=(
            "Face-expression assist is off (ENABLE_FACE_AGGRESSION=0). "
            "Faces are not analyzed. Enable only for authorized lab use; "
            "cues are unreliable and never identify a person."
        ),
    )


def _cascade_path() -> Optional[Path]:
    try:
        base = getattr(cv2, "data", None)
        if base is None:
            return None
        haarcascades = getattr(base, "haarcascades", "") or ""
        path = Path(haarcascades) / "haarcascade_frontalface_default.xml"
        if path.is_file():
            return path
    except Exception:
        return None
    return None


def analyze_face_aggression(
    image_bgr: np.ndarray,
    enabled: Optional[bool] = None,
) -> FaceAggressionResult:
    """Best-effort Haar face find + conservative geometric tension proxy.

    Geometric contrast in brow/mouth bands is **not** facial emotion
    recognition. Scores are capped and labeled assistive only.
    """
    if not face_aggression_enabled(enabled):
        return _disabled_result()

    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return FaceAggressionResult(
            status="none_detected",
            score=None,
            note="No image data — no face analyzed.",
            faces_found=0,
        )

    cascade_file = _cascade_path()
    if cascade_file is None:
        return FaceAggressionResult(
            status="unavailable",
            score=None,
            note=(
                "face_aggression=unavailable: OpenCV Haar cascade not found. "
                "No FER model is bundled (a robust one would require a large "
                "download). Faces were not scored."
            ),
            faces_found=0,
        )

    try:
        classifier = cv2.CascadeClassifier(str(cascade_file))
        if classifier.empty():
            raise RuntimeError("empty cascade")
        if image_bgr.ndim == 2:
            gray = image_bgr
        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24)
        )
    except Exception:
        return FaceAggressionResult(
            status="unavailable",
            score=None,
            note=(
                "face_aggression=unavailable: face detector failed to run. "
                "No scores invented."
            ),
            faces_found=0,
        )

    n_faces = 0 if faces is None else len(faces)
    if n_faces == 0:
        return FaceAggressionResult(
            status="none_detected",
            score=None,
            note="No face detected. Face cues were not applied.",
            faces_found=0,
        )

    # Conservative tension proxy: local contrast in brow + mouth bands.
    # Capped well below a confident claim. Assistive rationale only.
    x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    roi = gray[max(0, y) : y + h, max(0, x) : x + w]
    if roi.size == 0:
        proxy = 0.15
    else:
        hh, ww = roi.shape[:2]
        brow = roi[0 : max(1, hh // 3), :]
        mouth = roi[max(0, (2 * hh) // 3) : hh, :]
        brow_c = float(brow.std()) / 255.0 if brow.size else 0.0
        mouth_c = float(mouth.std()) / 255.0 if mouth.size else 0.0
        proxy = float(min(0.40, 0.12 + 0.55 * brow_c + 0.45 * mouth_c))

    if proxy >= 0.25:
        note = (
            "Possible tense facial expression — verify. "
            "Unreliable geometric proxy only; not identity, not emotion "
            "recognition, and not a criminal label."
        )
    else:
        note = (
            "Face visible; expression inconclusive. "
            "Assistive only — a human must verify. No identity attached."
        )
    return FaceAggressionResult(
        status="assistive",
        score=proxy,
        note=note,
        faces_found=n_faces,
    )
