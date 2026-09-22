"""Object / structure inventory assist.

YOLO (when ultralytics is installed) maps COCO boxes onto the curated
catalog. MOCK emits scripted demo objects **only** when callers opt in
(synthetic runs). Real uploads never invent a refrigerator without a
detector. Humans verify; this is not enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from vision.detector import Detection
from vision.objects_catalog import (
    ObjectEntry,
    map_detector_label,
    object_display_name,
)

logger = logging.getLogger(__name__)

UNAVAILABLE_NOTE = "Object detection is unavailable."
YOLO_NOTE = ""
MOCK_NOTE = ""


@dataclass
class CatalogObject:
    object_id: str
    display_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    source: str = "yolo"
    group: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.object_id,
            "display": self.display_name or object_display_name(self.object_id),
            "confidence": round(float(self.confidence), 3),
            "bbox_xyxy": [float(x) for x in self.bbox_xyxy],
            "source": self.source,
            "group": self.group,
        }

    def to_detection(self) -> Detection:
        return Detection(
            label=self.object_id,
            confidence=float(self.confidence),
            bbox_xyxy=self.bbox_xyxy,
            extras={
                "catalog_object_id": self.object_id,
                "catalog_display": self.display_name,
                "catalog_group": self.group,
                "object_source": self.source,
            },
        )


@dataclass
class ObjectInventory:
    backend: str = "unavailable"
    objects: List[CatalogObject] = field(default_factory=list)
    note: str = UNAVAILABLE_NOTE

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "objects": [obj.to_dict() for obj in self.objects],
            "note": self.note,
            "counts": _count_ids(self.objects),
        }

    @property
    def object_ids(self) -> list[str]:
        return [obj.object_id for obj in self.objects]


def _count_ids(objects: Sequence[CatalogObject]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in objects:
        counts[obj.object_id] = counts.get(obj.object_id, 0) + 1
    return counts


def _from_entry(
    entry: ObjectEntry,
    confidence: float,
    bbox: tuple[float, float, float, float],
    source: str,
) -> CatalogObject:
    return CatalogObject(
        object_id=entry.id,
        display_name=entry.display_name,
        confidence=float(confidence),
        bbox_xyxy=bbox,
        source=source,
        group=entry.group,
    )


def objects_from_detections(
    detections: Sequence[Detection],
    *,
    source: str = "adapter",
) -> List[CatalogObject]:
    """Keep detections whose labels map onto the catalog."""
    hits: List[CatalogObject] = []
    seen: set[tuple[str, tuple[float, float, float, float]]] = set()
    for det in detections:
        extras = det.extras or {}
        raw_id = extras.get("catalog_object_id")
        entry = map_detector_label(str(raw_id)) if raw_id else map_detector_label(det.label)
        if entry is None:
            continue
        bbox = tuple(float(x) for x in det.bbox_xyxy)
        key = (entry.id, bbox)
        if key in seen:
            continue
        seen.add(key)
        hits.append(_from_entry(entry, det.confidence, bbox, source))
    return hits


# Scripted furniture / civic objects for the synthetic MOCK demo only.
_MOCK_SCRIPT: dict[int, list[tuple[str, float, tuple[float, float, float, float]]]] = {
    0: [],
    1: [("chair", 0.88, (40.0, 220.0, 140.0, 400.0))],
    2: [
        ("table", 0.86, (160.0, 240.0, 360.0, 360.0)),
        ("chair", 0.84, (80.0, 230.0, 160.0, 400.0)),
    ],
    3: [
        ("refrigerator", 0.91, (20.0, 80.0, 140.0, 420.0)),
        ("sink", 0.80, (180.0, 260.0, 300.0, 340.0)),
    ],
    4: [
        ("bench", 0.87, (80.0, 280.0, 280.0, 380.0)),
        ("fence", 0.72, (0.0, 200.0, 640.0, 280.0)),
    ],
    5: [("television", 0.83, (200.0, 80.0, 420.0, 240.0))],
    6: [("playground_equipment", 0.78, (60.0, 120.0, 360.0, 400.0))],
    7: [("sofa", 0.85, (80.0, 220.0, 400.0, 400.0))],
    8: [("gate", 0.74, (200.0, 80.0, 360.0, 420.0))],
    9: [("fire_hydrant", 0.81, (40.0, 280.0, 90.0, 380.0))],
}


def mock_objects_for_frame(frame_index: int) -> List[CatalogObject]:
    key = int(frame_index) % max(1, len(_MOCK_SCRIPT))
    rows = _MOCK_SCRIPT.get(key, [])
    hits: List[CatalogObject] = []
    for object_id, conf, bbox in rows:
        entry = map_detector_label(object_id)
        if entry is None:
            continue
        hits.append(_from_entry(entry, conf, bbox, "mock"))
    return hits


_YOLO_MODEL = None
_YOLO_TRIED = False


def reset_yolo_cache() -> None:
    """Test helper — forget a previously loaded (or failed) YOLO model."""
    global _YOLO_MODEL, _YOLO_TRIED
    _YOLO_MODEL = None
    _YOLO_TRIED = False


def yolo_model_available() -> bool:
    return try_load_yolo() is not None


def try_load_yolo(model_name: str = "yolov8n.pt"):
    """Load ultralytics YOLO once. Never raises; None means unavailable."""
    global _YOLO_MODEL, _YOLO_TRIED
    if _YOLO_TRIED:
        return _YOLO_MODEL
    _YOLO_TRIED = True
    try:
        from ultralytics import YOLO  # type: ignore

        _YOLO_MODEL = YOLO(model_name)
        logger.info("Object inventory: loaded ultralytics YOLO (%s)", model_name)
    except Exception as exc:  # ImportError or weight/load failure
        logger.info("Object inventory: YOLO unavailable (%s)", exc)
        _YOLO_MODEL = None
    return _YOLO_MODEL


def detect_yolo_catalog_objects(image_bgr: np.ndarray) -> Optional[List[CatalogObject]]:
    """Run YOLO and keep catalog-overlapping boxes.

    Returns None when the backend cannot load (caller should report
    ``unavailable``). An empty list means the detector ran and saw nothing
    in the catalog — not an invitation to invent objects.
    """
    model = try_load_yolo()
    if model is None:
        return None
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return []
    try:
        results = model.predict(image_bgr, verbose=False)
    except Exception as exc:
        logger.info("Object inventory: YOLO predict failed (%s)", exc)
        return None
    if not results:
        return []
    result = results[0]
    names = result.names or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    hits: List[CatalogObject] = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = str(names.get(cls_id, f"class_{cls_id}")).lower()
        entry = map_detector_label(label)
        if entry is None:
            continue
        raw_xyxy = box.xyxy[0]
        xyxy = raw_xyxy.tolist() if hasattr(raw_xyxy, "tolist") else list(raw_xyxy)
        bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        hits.append(_from_entry(entry, conf, bbox, "yolo"))
    return hits


def collect_object_inventory(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
    *,
    frame_index: int = 0,
    mode: str = "auto",
    adapter_name: str = "",
) -> ObjectInventory:
    """Build a per-frame catalog inventory.

    ``mode``:
      * ``auto`` (default) — map adapter labels; else YOLO sidecar; else
        honest ``unavailable``. Never uses MOCK.
      * ``mock`` — scripted demo objects (synthetic pipeline only).
      * ``off`` — skip inventory.
    """
    choice = (mode or "auto").strip().lower()
    if choice in {"off", "none", "disabled"}:
        return ObjectInventory(
            backend="off",
            objects=[],
            note="Object inventory disabled for this run.",
        )

    adapter = (adapter_name or "").strip().lower()
    mapped_source = "yolo" if adapter == "yolo" else "adapter"
    mapped = objects_from_detections(detections, source=mapped_source)

    if choice == "mock":
        scripted = mock_objects_for_frame(frame_index)
        return ObjectInventory(backend="mock", objects=scripted, note=MOCK_NOTE)

    if adapter == "yolo":
        return ObjectInventory(backend="yolo", objects=mapped, note=YOLO_NOTE)

    if mapped:
        return ObjectInventory(
            backend=mapped_source,
            objects=mapped,
            note=YOLO_NOTE if mapped_source == "yolo" else (
                "Catalog objects mapped from detector labels. Assistive only."
            ),
        )

    yolo_hits = detect_yolo_catalog_objects(image_bgr)
    if yolo_hits is None:
        return ObjectInventory(
            backend="unavailable",
            objects=[],
            note=UNAVAILABLE_NOTE,
        )
    return ObjectInventory(backend="yolo", objects=yolo_hits, note=YOLO_NOTE)


def merge_inventory_detections(
    detections: Sequence[Detection],
    inventory: ObjectInventory,
) -> List[Detection]:
    """Append catalog objects that are not already present as detections."""
    merged = list(detections)
    existing = {
        (
            (d.extras or {}).get("catalog_object_id") or d.label,
            tuple(float(x) for x in d.bbox_xyxy),
        )
        for d in merged
    }
    for obj in inventory.objects:
        key = (obj.object_id, tuple(float(x) for x in obj.bbox_xyxy))
        if key in existing:
            continue
        merged.append(obj.to_detection())
        existing.add(key)
    return merged
