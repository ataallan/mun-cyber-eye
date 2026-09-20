"""CPU-only OpenCV features for Phase 3 activity recognition.

Handcrafted descriptors (color histogram, HOG, geometry, optional motion)
so training and inference run without a GPU. This is intentionally lightweight
and is not a production human-activity model.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

# Keep in lockstep with extract_frame_features(). Bump when the vector changes.
FEATURE_VERSION = 1
FEATURE_SIZE = 64  # width
FEATURE_HEIGHT = 48

# Gradient histograms: 64x48, 8x8 cells, 9 orientation bins → 8 x 6 x 9 = 432
_HOG_DIM = 432
_HIST_BINS = (6, 6, 4)  # H, S, V
_HIST_DIM = 6 * 6 * 4  # 144
_STAT_DIM = 16
FEATURE_DIM = _HIST_DIM + _HOG_DIM + _STAT_DIM  # 592


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("empty image")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def extract_frame_features(
    image_bgr: np.ndarray,
    prev_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a fixed-length float32 feature vector for one BGR frame.

    ``prev_bgr`` is an optional previous frame used for cheap motion stats.
    Single stills (demo dataset images) simply contribute zeros for motion.
    """
    img = cv2.resize(
        _as_bgr(image_bgr),
        (FEATURE_SIZE, FEATURE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        _HIST_BINS,
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

    hog_feat = _gradient_histograms(gray)

    edges = cv2.Canny(gray, 60, 140)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    # Largest-contour geometry: fall scenes tend to be wide/low; standing people tall.
    aspect = 0.5
    cy_norm = 0.5
    area_frac = 0.0
    n_blobs = 0.0
    contours, _ = cv2.findContours(
        (edges > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        significant = [c for c in contours if cv2.contourArea(c) >= 20]
        n_blobs = float(min(len(significant), 8)) / 8.0
        if significant:
            largest = max(significant, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            aspect = float(w) / float(h + 1e-6)
            aspect = float(np.clip(aspect / 4.0, 0.0, 1.0))  # normalize ~[0, 4]
            m = cv2.moments(largest)
            if m["m00"] > 0:
                cy_norm = float(m["m01"] / m["m00"]) / float(FEATURE_HEIGHT)
            area_frac = float(cv2.contourArea(largest)) / float(
                FEATURE_SIZE * FEATURE_HEIGHT
            )

    b, g, r = cv2.split(img)
    sat = hsv[:, :, 1]
    # Yellow-ish object proxy (weapon-like demo cue): high R+G, low B
    yellow = np.clip((r.astype(np.int16) + g.astype(np.int16)) / 2 - b.astype(np.int16), 0, 255)

    motion_mean = 0.0
    motion_std = 0.0
    motion_frac = 0.0
    if prev_bgr is not None and getattr(prev_bgr, "size", 0) > 0:
        prev = cv2.resize(
            _as_bgr(prev_bgr),
            (FEATURE_SIZE, FEATURE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_g)
        motion_mean = float(diff.mean()) / 255.0
        motion_std = float(diff.std()) / 255.0
        motion_frac = float((diff > 18).mean())

    stats = np.array(
        [
            float(edges.mean()) / 255.0,
            float(mag.mean()) / 255.0,
            float(mag.std()) / 255.0,
            aspect,
            cy_norm,
            area_frac,
            n_blobs,
            float(r.mean()) / 255.0,
            float(b.mean()) / 255.0,
            float(sat.mean()) / 255.0,
            float(yellow.mean()) / 255.0,
            float(gray.std()) / 255.0,
            motion_mean,
            motion_std,
            motion_frac,
            float(img.mean()) / 255.0,
        ],
        dtype=np.float32,
    )
    assert stats.size == _STAT_DIM

    vec = np.concatenate([hist, hog_feat, stats]).astype(np.float32)
    if vec.size != FEATURE_DIM:
        raise RuntimeError(f"feature dim {vec.size} != {FEATURE_DIM}")
    if not np.isfinite(vec).all():
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def _gradient_histograms(gray: np.ndarray, cell: int = 8, nbins: int = 9) -> np.ndarray:
    """HOG-like cell orientation histograms (no cv2.HOGDescriptor — OpenCV 4/5)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    h, w = gray.shape[:2]
    ncy, ncx = h // cell, w // cell
    parts: list[np.ndarray] = []
    for cy in range(ncy):
        for cx in range(ncx):
            m = mag[cy * cell : (cy + 1) * cell, cx * cell : (cx + 1) * cell].ravel()
            a = ang[cy * cell : (cy + 1) * cell, cx * cell : (cx + 1) * cell].ravel()
            hist, _ = np.histogram(a, bins=nbins, range=(0.0, 180.0), weights=m)
            total = float(hist.sum())
            if total > 0:
                hist = hist / total
            parts.append(hist.astype(np.float32))
    if not parts:
        return np.zeros(_HOG_DIM, dtype=np.float32)
    vec = np.concatenate(parts).astype(np.float32)
    if vec.size < _HOG_DIM:
        vec = np.pad(vec, (0, _HOG_DIM - int(vec.size)))
    return vec[:_HOG_DIM]


def extract_stack_features(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Features for a short frame stack (last frame + motion vs previous)."""
    if not frames:
        raise ValueError("frame stack is empty")
    prev = frames[-2] if len(frames) >= 2 else None
    return extract_frame_features(frames[-1], prev)
