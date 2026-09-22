"""Curated dangerous / weapon-like object catalog.

Assistive only. YOLO overlap is incomplete; toys, tools, and phones
false-fire. Never ballistic certainty, never a guilt label, never
enforcement. Humans verify.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from vision.dataset import IMAGE_SUFFIXES, SPLITS

HARM_LOW = "low"
HARM_MEDIUM = "medium"
HARM_HIGH = "high"

CLASS_EDGED = "edged"
CLASS_BLUNT = "blunt"
CLASS_FIREARM = "firearm_like"
CLASS_IMPROVISED = "improvised"
CLASS_CHEMICAL_FIRE = "chemical_fire"

USE_NONE = "none"
USE_BRANDISHED = "brandished"
USE_THREATENING = "threatening_motion"
USE_STRIKE = "possible_strike"

HONEST_NOTE = (
    "Dangerous-object labels are a curated assist — not proof of a real "
    "weapon, possession, or intent. Toys, tools, phones, and occlusion "
    "false-fire. Humans verify. The system does not enforce or dispatch."
)


@dataclass(frozen=True)
class DangerousEntry:
    id: str
    display_name: str
    weapon_class: str
    harm_potential: str
    aliases: tuple[str, ...] = ()
    sport_soften: tuple[str, ...] = ()
    context_dependent: bool = False
    notes: str = ""
    dangerous_object: bool = True


# Practical list. Not every SKU or improvised item on earth.
DANGEROUS_OBJECTS: tuple[DangerousEntry, ...] = (
    # --- Edged ---
    DangerousEntry(
        "knife",
        "Knife",
        CLASS_EDGED,
        HARM_HIGH,
        ("kitchen_knife", "blade", "steak_knife"),
        notes="YOLO/COCO overlap. Table knives and toys look the same.",
    ),
    DangerousEntry(
        "machete",
        "Machete",
        CLASS_EDGED,
        HARM_HIGH,
        ("cutlass",),
        notes="No COCO class — needs operator-labeled frames.",
    ),
    DangerousEntry(
        "scissors",
        "Scissors (proxy)",
        CLASS_EDGED,
        HARM_MEDIUM,
        ("shears",),
        notes="COCO scissors is a weak edged proxy — often craft/office use.",
    ),
    # --- Blunt ---
    DangerousEntry(
        "baseball_bat",
        "Baseball bat",
        CLASS_BLUNT,
        HARM_MEDIUM,
        ("baseball bat", "bat", "cricket_bat"),
        sport_soften=("baseball", "softball", "cricket"),
        notes="Sport equipment. Low use-intensity on a matching field may soften.",
    ),
    DangerousEntry(
        "crowbar",
        "Crowbar",
        CLASS_BLUNT,
        HARM_HIGH,
        ("prybar", "pry_bar"),
        notes="No COCO class — operator frames required.",
    ),
    DangerousEntry(
        "metal_pipe",
        "Metal pipe",
        CLASS_BLUNT,
        HARM_HIGH,
        ("pipe", "steel_pipe"),
        notes="No COCO class. Plumbing in frame is a false positive risk.",
    ),
    DangerousEntry(
        "heavy_stick",
        "Heavy stick / club",
        CLASS_BLUNT,
        HARM_MEDIUM,
        ("club", "baton", "stick"),
        notes="Very weak visual proxy.",
    ),
    # --- Firearm-like (visual proxy only) ---
    DangerousEntry(
        "handgun",
        "Handgun (visual proxy)",
        CLASS_FIREARM,
        HARM_HIGH,
        ("gun", "pistol", "firearm", "hand_gun"),
        notes="Never ballistic certainty. Phones, toys, and occlusion fool this.",
    ),
    DangerousEntry(
        "rifle",
        "Rifle (visual proxy)",
        CLASS_FIREARM,
        HARM_HIGH,
        ("long_gun", "shotgun", "longarm"),
        notes="Never ballistic certainty. Umbrellas and tools false-fire.",
    ),
    # --- Improvised (context-dependent) ---
    DangerousEntry(
        "bottle",
        "Bottle (improvised)",
        CLASS_IMPROVISED,
        HARM_MEDIUM,
        ("wine glass", "wine_glass", "beer_bottle"),
        context_dependent=True,
        notes="A bottle on a table is not a weapon. Raised / thrown / strike cues required.",
    ),
    DangerousEntry(
        "brick",
        "Brick (improvised)",
        CLASS_IMPROVISED,
        HARM_HIGH,
        ("rock", "stone"),
        context_dependent=True,
        notes="No COCO brick class. Do not invent from a wall.",
    ),
    DangerousEntry(
        "chair_as_weapon",
        "Chair used as weapon",
        CLASS_IMPROVISED,
        HARM_MEDIUM,
        (),
        context_dependent=True,
        notes="Ordinary chairs stay furniture. Only when raised / strike-like toward a person.",
    ),
    # --- Optional chemical / fire (honest, weak) ---
    DangerousEntry(
        "spray_can",
        "Aerosol / spray can (proxy)",
        CLASS_CHEMICAL_FIRE,
        HARM_LOW,
        ("aerosol", "spray_bottle"),
        context_dependent=True,
        notes="Not a chemical-weapon classifier. Hair spray and paint cans look the same.",
    ),
    DangerousEntry(
        "fire_proxy",
        "Fire / flame (visual proxy)",
        CLASS_CHEMICAL_FIRE,
        HARM_MEDIUM,
        ("fire", "flame", "molotov"),
        context_dependent=True,
        notes="Color/bright-region proxy only — not an accelerant or arson determination.",
    ),
    DangerousEntry(
        "unidentified_improvised",
        "Unidentified striking object",
        CLASS_IMPROVISED,
        HARM_HIGH,
        (
            "improvised_weapon",
            "unknown_weapon",
            "unidentified_object",
            "unidentified_striking_object",
        ),
        context_dependent=True,
        notes=(
            "Unknown or not-in-catalog object used to hit or thrown at a person. "
            "Not a named weapon. Needs labeled frames. Presence alone is not a cue."
        ),
    ),
)


# COCO / YOLO labels that overlap this catalog. Everything else needs frames.
YOLO_KNOWN_IDS = frozenset({"knife", "scissors", "baseball_bat", "bottle"})


def _norm(name: str) -> str:
    return (
        (name or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _build_maps() -> tuple[dict[str, DangerousEntry], dict[str, str]]:
    by_id: dict[str, DangerousEntry] = {}
    alias: dict[str, str] = {}
    for entry in DANGEROUS_OBJECTS:
        by_id[entry.id] = entry
        alias[_norm(entry.id)] = entry.id
        alias[_norm(entry.display_name)] = entry.id
        for raw in entry.aliases:
            alias[_norm(raw)] = entry.id
            alias[raw.strip().lower()] = entry.id
    return by_id, alias


_BY_ID, _ALIAS = _build_maps()

# Labels that are dedicated weapon-like (presence can alert).
DEDICATED_LABELS = frozenset(
    entry.id for entry in DANGEROUS_OBJECTS if not entry.context_dependent
)
CONTEXT_LABELS = frozenset(
    entry.id for entry in DANGEROUS_OBJECTS if entry.context_dependent
)


def all_dangerous_objects() -> tuple[DangerousEntry, ...]:
    return DANGEROUS_OBJECTS


def get_dangerous(object_id: str) -> Optional[DangerousEntry]:
    return _BY_ID.get(_norm(object_id)) or _BY_ID.get(_ALIAS.get(_norm(object_id), ""))


def map_detector_label(label: str) -> Optional[DangerousEntry]:
    """Map a detector / YOLO / alias label onto the dangerous catalog."""
    raw = (label or "").strip()
    if not raw:
        return None
    key = _norm(raw)
    oid = _ALIAS.get(key) or _ALIAS.get(raw.lower())
    if oid:
        return _BY_ID[oid]
    # "baseball bat" with space is stored as an alias too.
    oid = _ALIAS.get(raw.lower().replace(" ", "_"))
    return _BY_ID.get(oid) if oid else None


def dangerous_display_name(object_id: str) -> str:
    entry = get_dangerous(object_id)
    if entry is None:
        return (object_id or "").replace("_", " ").strip() or "unknown"
    return entry.display_name


def validate_dangerous_id(object_id: str) -> str:
    raw = (object_id or "").strip()
    if ".." in raw or "/" in raw or "\\" in raw:
        raise ValueError(f"Unknown dangerous-object class '{object_id}'.")
    entry = map_detector_label(raw) or get_dangerous(raw)
    if entry is None:
        raise ValueError(
            f"Unknown dangerous-object class '{object_id}'. "
            "Use a catalog id such as knife, baseball_bat, handgun, bottle."
        )
    return entry.id


def detector_signal_labels() -> frozenset[str]:
    """Labels the risk engine may treat as dedicated weapon-like hits."""
    labels = set()
    for entry in DANGEROUS_OBJECTS:
        if entry.context_dependent:
            continue
        labels.add(entry.id)
        labels.update(a.lower() for a in entry.aliases)
        labels.add(entry.display_name.lower())
    labels.update(
        {
            "raised_object",
            "suspicious_object",
            "firearm_aimed_at_person",
            "weapon_pointed_at_person",
            "unidentified_striking_object",
        }
    )
    return frozenset(labels)


def chair_is_weapon_label(label: str) -> bool:
    return _norm(label) in {"chair_as_weapon"} or _norm(label).startswith("chair_as")


def ensure_dangerous_tree(
    root: str | Path, object_ids: Iterable[str] | None = None
) -> Path:
    root_path = Path(root)
    ids = list(object_ids) if object_ids is not None else [e.id for e in DANGEROUS_OBJECTS]
    for split in SPLITS:
        for oid in ids:
            (root_path / split / oid).mkdir(parents=True, exist_ok=True)
    return root_path


def describe_dangerous_dataset(root: str | Path) -> dict:
    """Count labeled frames under data/dangerous/{split}/{id}/."""
    root_path = Path(root)
    splits: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    totals = {entry.id: 0 for entry in DANGEROUS_OBJECTS}
    extra: dict[str, int] = {}
    for split in SPLITS:
        split_dir = root_path / split
        if not split_dir.is_dir():
            for entry in DANGEROUS_OBJECTS:
                splits[split][entry.id] = 0
            continue
        counted: set[str] = set()
        for child in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if not child.is_dir():
                continue
            n = sum(
                1
                for p in child.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            entry = get_dangerous(child.name) or map_detector_label(child.name)
            key = entry.id if entry is not None else child.name
            splits[split][key] = splits[split].get(key, 0) + n
            if entry is not None:
                totals[entry.id] = totals.get(entry.id, 0) + n
            else:
                extra[key] = extra.get(key, 0) + n
            counted.add(key)
        for entry in DANGEROUS_OBJECTS:
            if entry.id not in counted:
                splits[split][entry.id] = splits[split].get(entry.id, 0)
    labeled = {oid: n for oid, n in totals.items() if n > 0}
    n_images = sum(totals.values()) + sum(extra.values())
    return {
        "root": str(root_path),
        "counts": {split: {k: v for k, v in folders.items() if v} for split, folders in splits.items()},
        "splits": splits,
        "totals": totals,
        "labeled": labeled,
        "extra_folders": extra,
        "n_classes_with_images": len(labeled),
        "n_images": n_images,
        "total_images": n_images,
    }


def weapon_should_soften_sport(
    *,
    weapon_id: str,
    sport_context: Optional[str],
    sports_venue: bool,
    sport_decent: bool,
    use_intensity: float,
    use_intensity_label: str,
    aimed_at_person: bool = False,
    gunshot_proxy: bool = False,
) -> bool:
    """Bat on a matching field with low use-intensity may stay play."""
    if aimed_at_person or gunshot_proxy:
        return False
    if use_intensity_label == USE_STRIKE or float(use_intensity) >= 0.80:
        return False
    entry = map_detector_label(weapon_id) or get_dangerous(weapon_id)
    if entry is None or not entry.sport_soften:
        return False
    sport = (sport_context or "").strip().lower()
    if sport not in entry.sport_soften:
        return False
    if not (sports_venue or sport_decent):
        return False
    if use_intensity_label in {USE_THREATENING} and float(use_intensity) >= 0.70:
        return False
    return float(use_intensity) < 0.65 or use_intensity_label in {
        USE_NONE,
        USE_BRANDISHED,
        "",
    }
