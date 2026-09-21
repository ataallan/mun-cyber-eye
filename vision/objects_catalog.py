"""Curated home / community object and structure catalog.

This is an **assistive inventory**, not a claim that the pipeline can
recognize every appliance, piece of furniture, or civic structure. YOLO
(COCO) covers a subset; unlabeled catalog ids need operator frames.
Humans verify. The system does not enforce, detain, or lock anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from vision.dataset import IMAGE_SUFFIXES, SPLITS


@dataclass(frozen=True)
class ObjectEntry:
    id: str
    display_name: str
    group: str  # home | community
    aliases: tuple[str, ...] = ()
    notes: str = ""


# Practical list (~50). Not infinite; operators can label site-specific
# examples into these folders rather than inventing new ids ad hoc.
OBJECTS: tuple[ObjectEntry, ...] = (
    # --- Home / indoor equipment ---
    ObjectEntry("refrigerator", "Refrigerator", "home", ("fridge", "freezer")),
    ObjectEntry("chair", "Chair", "home", ("armchair", "stool")),
    ObjectEntry("table", "Table", "home", ("dining_table", "diningtable", "kitchen_table")),
    ObjectEntry("sofa", "Sofa / couch", "home", ("couch", "settee", "loveseat")),
    ObjectEntry("bed", "Bed", "home", ()),
    ObjectEntry("desk", "Desk", "home", ("work_desk",)),
    ObjectEntry("television", "Television", "home", ("tv", "tvmonitor", "monitor_tv")),
    ObjectEntry("microwave", "Microwave", "home", ()),
    ObjectEntry("oven", "Oven / stove", "home", ("stove", "cooker", "range")),
    ObjectEntry("sink", "Sink", "home", ("kitchen_sink", "bathroom_sink")),
    ObjectEntry("toilet", "Toilet", "home", ()),
    ObjectEntry("door", "Door", "home", ()),
    ObjectEntry("window", "Window", "home", ()),
    ObjectEntry("lamp", "Lamp", "home", ("floor_lamp", "table_lamp")),
    ObjectEntry("bookshelf", "Bookshelf", "home", ("bookcase",)),
    ObjectEntry("wardrobe", "Wardrobe", "home", ("closet", "armoire")),
    ObjectEntry("washing_machine", "Washing machine", "home", ("washer", "laundry_machine")),
    ObjectEntry("dishwasher", "Dishwasher", "home", ()),
    ObjectEntry("coffee_maker", "Coffee maker", "home", ("coffee_machine", "espresso_machine")),
    ObjectEntry("toaster", "Toaster", "home", ()),
    ObjectEntry("bathtub", "Bathtub", "home", ("tub",)),
    ObjectEntry("shower", "Shower", "home", ()),
    ObjectEntry("nightstand", "Nightstand", "home", ("bedside_table",)),
    ObjectEntry("dresser", "Dresser", "home", ("chest_of_drawers", "drawers")),
    ObjectEntry("mirror", "Mirror", "home", ()),
    ObjectEntry("cabinet", "Cabinet", "home", ("cupboard",)),
    ObjectEntry("counter", "Counter", "home", ("countertop", "kitchen_counter")),
    ObjectEntry("fan", "Fan", "home", ("ceiling_fan",)),
    ObjectEntry("heater", "Heater", "home", ("radiator", "space_heater")),
    ObjectEntry("air_conditioner", "Air conditioner", "home", ("ac_unit", "hvac_unit")),
    ObjectEntry("computer", "Computer", "home", ("laptop", "desktop_computer", "pc")),
    # --- Community / outdoor structures ---
    ObjectEntry("bench", "Bench", "community", ("park_bench",)),
    ObjectEntry("fence", "Fence", "community", ()),
    ObjectEntry("gate", "Gate", "community", ("entrance_gate",)),
    ObjectEntry(
        "playground_equipment",
        "Playground equipment",
        "community",
        ("swing_set", "slide", "jungle_gym", "play_structure"),
    ),
    ObjectEntry("trash_bin", "Trash bin", "community", ("garbage_can", "rubbish_bin", "waste_bin")),
    ObjectEntry("street_light", "Street light", "community", ("lamppost", "streetlamp")),
    ObjectEntry("fire_hydrant", "Fire hydrant", "community", ("hydrant",)),
    ObjectEntry("mailbox", "Mailbox", "community", ("postbox", "letterbox")),
    ObjectEntry("bus_stop", "Bus stop", "community", ("bus_shelter", "transit_stop")),
    ObjectEntry("picnic_table", "Picnic table", "community", ()),
    ObjectEntry("traffic_light", "Traffic light", "community", ("stoplight", "traffic_signal")),
    ObjectEntry("stop_sign", "Stop sign", "community", ()),
    ObjectEntry("parking_meter", "Parking meter", "community", ()),
    ObjectEntry("bicycle_rack", "Bicycle rack", "community", ("bike_rack",)),
    ObjectEntry("water_fountain", "Water fountain", "community", ("drinking_fountain",)),
    ObjectEntry("planter", "Planter / potted plant", "community", ("potted_plant", "plant_pot")),
    ObjectEntry("dumpster", "Dumpster", "community", ("skip",)),
    ObjectEntry("kiosk", "Kiosk", "community", ()),
    ObjectEntry("outdoor_sign", "Outdoor sign", "community", ("wayfinding_sign",)),
    ObjectEntry("bollard", "Bollard", "community", ()),
    ObjectEntry("flagpole", "Flagpole", "community", ()),
)


# COCO / Ultralytics YOLO label → catalog id. Only overlapping classes.
# Do not map person, knife, sports ball, or food onto furniture.
COCO_TO_CATALOG: dict[str, str] = {
    "chair": "chair",
    "couch": "sofa",
    "sofa": "sofa",
    "bed": "bed",
    "dining table": "table",
    "dining_table": "table",
    "tv": "television",
    "tvmonitor": "television",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "microwave": "microwave",
    "oven": "oven",
    "toaster": "toaster",
    "sink": "sink",
    "toilet": "toilet",
    "bench": "bench",
    "fire hydrant": "fire_hydrant",
    "fire_hydrant": "fire_hydrant",
    "potted plant": "planter",
    "potted_plant": "planter",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "stop sign": "stop_sign",
    "stop_sign": "stop_sign",
    "parking meter": "parking_meter",
    "parking_meter": "parking_meter",
    "laptop": "computer",
}


# Weak scene prior only. Chair/table are too generic to move place type.
HOME_PRIOR_IDS = frozenset(
    {
        "refrigerator",
        "sink",
        "oven",
        "microwave",
        "toilet",
        "bed",
        "sofa",
        "washing_machine",
        "dishwasher",
        "bathtub",
        "shower",
        "wardrobe",
        "nightstand",
        "dresser",
        "coffee_maker",
        "toaster",
        "cabinet",
        "counter",
    }
)
COMMUNITY_PRIOR_IDS = frozenset(
    {
        "bench",
        "fence",
        "gate",
        "playground_equipment",
        "trash_bin",
        "street_light",
        "fire_hydrant",
        "mailbox",
        "bus_stop",
        "picnic_table",
        "traffic_light",
        "stop_sign",
        "parking_meter",
        "bollard",
        "dumpster",
    }
)
OBJECT_PLACE_PRIOR: dict[str, str] = {
    "refrigerator": "house_interior",
    "sink": "house_interior",
    "oven": "house_interior",
    "microwave": "house_interior",
    "toilet": "house_interior",
    "bed": "house_interior",
    "sofa": "house_interior",
    "washing_machine": "house_interior",
    "dishwasher": "house_interior",
    "bathtub": "house_interior",
    "shower": "house_interior",
    "wardrobe": "house_interior",
    "nightstand": "house_interior",
    "dresser": "house_interior",
    "coffee_maker": "house_interior",
    "toaster": "house_interior",
    "cabinet": "house_interior",
    "counter": "house_interior",
    "playground_equipment": "playground",
    "bench": "outdoor_plaza",
    "picnic_table": "outdoor_plaza",
    "gate": "compound_courtyard",
    "fence": "compound_courtyard",
    "street_light": "street",
    "fire_hydrant": "street",
    "bus_stop": "street",
    "traffic_light": "street",
    "stop_sign": "street",
    "parking_meter": "street",
    "mailbox": "street",
    "bollard": "street",
    "dumpster": "outdoor_plaza",
    "trash_bin": "outdoor_plaza",
}

# Whole-frame sklearn object training is assistive and CPU-only. At least
# two classes with a few images each — never invent detections at runtime.
MIN_OBJECT_TRAIN_CLASSES = 2
MIN_OBJECT_TRAIN_IMAGES = 3


def _norm(name: str) -> str:
    return (
        (name or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def all_objects() -> tuple[ObjectEntry, ...]:
    return OBJECTS


def objects_by_group() -> dict[str, tuple[ObjectEntry, ...]]:
    groups: dict[str, list[ObjectEntry]] = {}
    for entry in OBJECTS:
        groups.setdefault(entry.group, []).append(entry)
    return {key: tuple(vals) for key, vals in groups.items()}


def object_alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in OBJECTS:
        mapping[_norm(entry.id)] = entry.id
        mapping[_norm(entry.display_name)] = entry.id
        for alias in entry.aliases:
            mapping[_norm(alias)] = entry.id
    return mapping


_ALIAS_MAP = object_alias_map()
_BY_ID = {entry.id: entry for entry in OBJECTS}


def get_object(object_id: str) -> Optional[ObjectEntry]:
    return _BY_ID.get(_norm(object_id))


def resolve_object(name: str) -> Optional[ObjectEntry]:
    """Resolve a catalog id, display name, or alias."""
    key = _norm(name)
    if not key:
        return None
    object_id = _ALIAS_MAP.get(key)
    if object_id is None:
        return None
    return _BY_ID[object_id]


def object_display_name(object_id: str) -> str:
    entry = get_object(object_id)
    if entry is None:
        return (object_id or "").replace("_", " ").strip() or "unknown"
    return entry.display_name


def validate_object_id(object_id: str) -> str:
    raw = (object_id or "").strip()
    if not raw:
        raise ValueError("Choose an object class from the catalog.")
    entry = resolve_object(raw)
    if entry is None:
        raise ValueError(
            f"Unknown object class '{object_id}'. "
            "Use a catalog id from vision/objects_catalog.py."
        )
    return entry.id


def map_detector_label(label: str) -> Optional[ObjectEntry]:
    """Map a YOLO/COCO (or already-catalog) label onto a catalog entry."""
    raw = (label or "").strip().lower()
    if not raw:
        return None
    mapped = COCO_TO_CATALOG.get(raw) or COCO_TO_CATALOG.get(raw.replace(" ", "_"))
    if mapped:
        return _BY_ID[mapped]
    return resolve_object(raw)


def ensure_objects_tree(root: str | Path, object_ids: Iterable[str] | None = None) -> Path:
    root_path = Path(root)
    ids = list(object_ids) if object_ids is not None else [e.id for e in OBJECTS]
    for split in SPLITS:
        for object_id in ids:
            (root_path / split / object_id).mkdir(parents=True, exist_ok=True)
    return root_path


def describe_objects_dataset(root: str | Path) -> dict:
    """Count labeled frames under data/objects/{split}/{object_id}/."""
    root_path = Path(root)
    splits: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    totals = {entry.id: 0 for entry in OBJECTS}
    extra: dict[str, int] = {}
    for split in SPLITS:
        split_dir = root_path / split
        if not split_dir.is_dir():
            for entry in OBJECTS:
                splits[split][entry.id] = 0
            continue
        counted: set[str] = set()
        for child in sorted(split_dir.iterdir()):
            if not child.is_dir():
                continue
            n = sum(
                1
                for p in child.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            entry = resolve_object(child.name)
            key = entry.id if entry is not None else child.name
            splits[split][key] = splits[split].get(key, 0) + n
            if entry is not None:
                totals[entry.id] = totals.get(entry.id, 0) + n
            else:
                extra[key] = extra.get(key, 0) + n
            counted.add(key)
        for entry in OBJECTS:
            if entry.id not in counted:
                splits[split][entry.id] = splits[split].get(entry.id, 0)
    labeled = {oid: n for oid, n in totals.items() if n > 0}
    return {
        "root": str(root_path),
        "splits": splits,
        "totals": totals,
        "labeled": labeled,
        "extra_folders": extra,
        "n_classes_with_images": len(labeled),
        "n_images": sum(totals.values()) + sum(extra.values()),
    }
