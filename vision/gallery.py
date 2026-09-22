"""Objects gallery: catalog cards and an honest scan against them.

Browse data is counts and sample filenames. Scan uses the YOLO sidecar
when it loads. A missing detector is ``unavailable`` and does not invent
matches. Humans verify.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from vision.dangerous_objects import (
    YOLO_KNOWN_IDS,
    all_dangerous_objects,
    map_detector_label as map_dangerous,
    validate_dangerous_id,
)
from vision.dataset import IMAGE_SUFFIXES, SPLITS
from vision.object_inventory import UNAVAILABLE_NOTE, try_load_yolo
from vision.objects_catalog import (
    COCO_TO_CATALOG,
    all_objects,
    map_detector_label as map_object,
    validate_object_id,
)

logger = logging.getLogger(__name__)

_YOLO_OBJECT_IDS = frozenset(COCO_TO_CATALOG.values())


def yolo_known(kind: str, object_id: str) -> bool:
    if kind == "dangerous":
        return object_id in YOLO_KNOWN_IDS
    return object_id in _YOLO_OBJECT_IDS


def parse_gallery_item(token: str) -> tuple[str, str]:
    raw = (token or "").strip()
    if ":" not in raw:
        raise ValueError("Choose a gallery item.")
    kind, object_id = raw.split(":", 1)
    kind = kind.strip().lower()
    if kind == "object":
        return kind, validate_object_id(object_id)
    if kind == "dangerous":
        return kind, validate_dangerous_id(object_id)
    raise ValueError("Choose a gallery item.")


def _sample_images(root: Path, object_id: str, limit: int = 2) -> list[dict]:
    found: list[dict] = []
    if not root:
        return found
    for split in SPLITS:
        folder = root / split / object_id
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            found.append({"split": split, "filename": path.name})
            if len(found) >= limit:
                return found
    return found


def list_gallery_cards(
    objects_root: str | Path,
    dangerous_root: str | Path,
) -> list[dict]:
    """Cards for home/community classes and dangerous classes."""
    from vision.dangerous_objects import describe_dangerous_dataset
    from vision.objects_catalog import describe_objects_dataset

    objects_path = Path(objects_root)
    dangerous_path = Path(dangerous_root)
    objects_info = describe_objects_dataset(objects_path)
    dangerous_info = describe_dangerous_dataset(dangerous_path)
    cards: list[dict] = []
    for entry in all_objects():
        train_count = int((objects_info.get("splits") or {}).get("train", {}).get(entry.id, 0) or 0)
        cards.append(
            {
                "kind": "object",
                "id": entry.id,
                "name": entry.display_name,
                "group": entry.group,
                "group_label": "Home" if entry.group == "home" else "Community",
                "train_count": train_count,
                "example_count": int((objects_info.get("totals") or {}).get(entry.id, 0) or 0),
                "yolo_known": yolo_known("object", entry.id),
                "thumbs": _sample_images(objects_path, entry.id),
                "token": f"object:{entry.id}",
            }
        )
    group_labels = {
        "edged": "Edged",
        "blunt": "Blunt",
        "firearm_like": "Firearm-like",
        "improvised": "Improvised",
        "chemical_fire": "Chemical / fire",
    }
    for entry in all_dangerous_objects():
        train_count = int(
            (dangerous_info.get("splits") or {}).get("train", {}).get(entry.id, 0) or 0
        )
        cards.append(
            {
                "kind": "dangerous",
                "id": entry.id,
                "name": entry.display_name,
                "group": "dangerous",
                "weapon_class": entry.weapon_class,
                "group_label": group_labels.get(entry.weapon_class, entry.weapon_class),
                "train_count": train_count,
                "example_count": int((dangerous_info.get("totals") or {}).get(entry.id, 0) or 0),
                "yolo_known": yolo_known("dangerous", entry.id),
                "thumbs": _sample_images(dangerous_path, entry.id),
                "token": f"dangerous:{entry.id}",
            }
        )
    return cards


def detect_gallery_labels(image_bgr: np.ndarray) -> Optional[list[tuple[str, float]]]:
    """YOLO labels for one frame.

    ``None`` means the detector did not load. An empty list means it ran
    and returned no boxes. Never invents a class.
    """
    model = try_load_yolo()
    if model is None:
        return None
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return []
    try:
        results = model.predict(image_bgr, verbose=False)
    except Exception as exc:
        logger.info("Gallery scan: YOLO predict failed (%s)", exc)
        return None
    if not results:
        return []
    result = results[0]
    names = result.names or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    labels: list[tuple[str, float]] = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = str(names.get(cls_id, "")).strip().lower()
        if not label:
            continue
        labels.append((label, conf))
    return labels


def matches_from_labels(pairs: Sequence[tuple[str, float]]) -> list[dict]:
    """Keep labels that map onto a gallery card. Drop everything else."""
    agg: dict[tuple[str, str], dict] = {}
    for label, confidence in pairs:
        dangerous = map_dangerous(label)
        obj = map_object(label)
        if dangerous is not None:
            key = ("dangerous", dangerous.id)
            name = dangerous.display_name
            group = "dangerous"
        elif obj is not None:
            key = ("object", obj.id)
            name = obj.display_name
            group = obj.group
        else:
            continue
        slot = agg.get(key)
        if slot is None:
            slot = {
                "kind": key[0],
                "id": key[1],
                "name": name,
                "group": group,
                "count": 0,
                "confidence": 0.0,
            }
            agg[key] = slot
        slot["count"] += 1
        try:
            slot["confidence"] = max(float(slot["confidence"]), float(confidence))
        except (TypeError, ValueError):
            pass
    rows = list(agg.values())
    rows.sort(key=lambda row: (-int(row["count"]), row["name"]))
    for row in rows:
        row["confidence"] = round(float(row["confidence"]), 3)
    return rows


def scan_frames(images: Sequence[np.ndarray]) -> dict:
    """Inventory a few frames and match them to gallery cards."""
    frames = [img for img in images if img is not None and getattr(img, "size", 0) > 0]
    if not frames:
        return {
            "backend": "unavailable",
            "matches": [],
            "note": "No frames to scan.",
            "frames": 0,
        }
    saw_detector = False
    pairs: list[tuple[str, float]] = []
    for image in frames:
        labels = detect_gallery_labels(image)
        if labels is None:
            continue
        saw_detector = True
        pairs.extend(labels)
    if not saw_detector:
        return {
            "backend": "unavailable",
            "matches": [],
            "note": UNAVAILABLE_NOTE,
            "frames": len(frames),
        }
    matches = matches_from_labels(pairs)
    note = "" if matches else "Detector ran. No gallery matches."
    return {
        "backend": "yolo",
        "matches": matches,
        "note": note,
        "frames": len(frames),
    }
