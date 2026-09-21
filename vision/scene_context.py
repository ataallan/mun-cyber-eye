"""Structured scene-context assist: place types and kit-color proxies.

This is **not** full scene understanding. The OpenCV + sklearn stack cannot
recognize every sports arena, street, or home. Operators get a catalog of
place types, optional camera stamps, folder tags, and honest synthetic
heuristics — never a famous venue name, a face identity, or a guilt label
from clothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from vision.detector import Detection
from vision.sports_catalog import resolve_sport

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PlaceEntry:
    id: str
    display_name: str
    group: str
    aliases: tuple[str, ...] = ()
    notes: str = ""


# Stable catalog (~27). Display names are generic — never a specific arena.
PLACES: tuple[PlaceEntry, ...] = (
    PlaceEntry("sports_field", "Sports field", "sports", ("field", "pitch", "soccer_pitch", "football_pitch")),
    PlaceEntry("basketball_court", "Basketball court", "sports", ("hoops_court", "court_basketball")),
    PlaceEntry("tennis_court", "Tennis court", "sports", ("court_tennis",)),
    PlaceEntry("volleyball_court", "Volleyball court", "sports", ("court_volleyball",)),
    PlaceEntry("indoor_arena", "Indoor arena", "sports", ("arena", "indoor_sports_arena")),
    PlaceEntry("gymnasium", "Gymnasium", "sports", ("gym", "school_gym")),
    PlaceEntry("playground", "Playground", "sports", ("play_area", "recess_yard")),
    PlaceEntry("track", "Track", "sports", ("running_track", "athletics_track")),
    PlaceEntry("swimming_pool", "Swimming pool", "sports", ("pool", "aquatic_center")),
    PlaceEntry("skate_park", "Skate park", "sports", ("skatepark", "skate_plaza")),
    PlaceEntry("street", "Street", "circulation", ("road", "roadway")),
    PlaceEntry("sidewalk", "Sidewalk", "circulation", ("pavement", "footpath")),
    PlaceEntry("parking_lot", "Parking lot", "circulation", ("parking", "car_park")),
    PlaceEntry("corridor_hallway", "Corridor / hallway", "circulation", ("corridor", "hallway", "hall")),
    PlaceEntry("lobby", "Lobby", "circulation", ("foyer", "entrance_lobby")),
    PlaceEntry("stairwell", "Stairwell", "circulation", ("stairs", "staircase")),
    PlaceEntry(
        "roam",
        "Roam / patrol (multi-area)",
        "circulation",
        ("roaming", "patrol", "mobile_camera"),
        "Mobile / patrol camera covering multiple areas. Circulation family — not a sports venue.",
    ),
    PlaceEntry("house_interior", "House / indoor home", "residential", ("house", "home", "indoor_home", "living_room")),
    PlaceEntry("residential_yard", "Residential yard", "residential", ("yard", "backyard", "garden")),
    PlaceEntry("compound_courtyard", "Compound / courtyard", "residential", ("compound", "courtyard")),
    PlaceEntry("driveway", "Driveway", "residential", ("drive",)),
    PlaceEntry("classroom_or_office", "Classroom or office", "other", ("classroom", "office", "classroom_office")),
    PlaceEntry("cafeteria", "Cafeteria", "other", ("canteen", "dining_hall")),
    PlaceEntry("warehouse_or_industrial", "Warehouse / industrial", "other", ("warehouse", "industrial")),
    PlaceEntry("parking_garage", "Parking garage", "other", ("garage",)),
    PlaceEntry("outdoor_plaza", "Outdoor plaza", "other", ("plaza",)),
    PlaceEntry("unknown", "Unknown", "other", ()),
)

SPORTS_VENUE_IDS = frozenset(p.id for p in PLACES if p.group == "sports")
CIRCULATION_IDS = frozenset(p.id for p in PLACES if p.group == "circulation")
RESIDENTIAL_IDS = frozenset(p.id for p in PLACES if p.group == "residential")

# Street / corridor / house / compound / roam — stronger lean toward
# confrontation when body-aggression is high and there is no sport context.
# Roam is circulation (patrol / multi-area), not a sports-venue soften.
STRONG_CONFRONTATION_IDS = frozenset(
    {
        "street",
        "corridor_hallway",
        "house_interior",
        "compound_courtyard",
        "roam",
    }
)
CONFRONTATION_SETTING_IDS = frozenset(
    {
        "street",
        "sidewalk",
        "parking_lot",
        "corridor_hallway",
        "stairwell",
        "roam",
        "house_interior",
        "residential_yard",
        "compound_courtyard",
        "driveway",
    }
)

# Catalog sport → default place. Assistive only; not a named stadium.
SPORT_DEFAULT_PLACE: dict[str, str] = {
    "basketball": "basketball_court",
    "soccer": "sports_field",
    "american_football": "sports_field",
    "volleyball": "volleyball_court",
    "tennis": "tennis_court",
    "badminton": "indoor_arena",
    "baseball": "sports_field",
    "softball": "sports_field",
    "cricket": "sports_field",
    "rugby": "sports_field",
    "hockey": "indoor_arena",
    "track_athletics": "track",
    "wrestling": "gymnasium",
    "boxing": "indoor_arena",
    "martial_arts_training": "gymnasium",
    "playground_games": "playground",
    "table_tennis": "gymnasium",
    "gymnastics": "gymnasium",
    "swimming": "swimming_pool",
    "skateboarding": "skate_park",
    "lacrosse": "sports_field",
    "golf": "sports_field",
    "cycling": "street",
    "pickleball": "tennis_court",
    "ultimate_frisbee": "sports_field",
    "water_polo": "swimming_pool",
    "fencing": "gymnasium",
    "climbing": "gymnasium",
    "dodgeball": "gymnasium",
    "kickball": "playground",
    "handball": "indoor_arena",
    "squash": "indoor_arena",
    "netball": "indoor_arena",
    "cheer": "indoor_arena",
}

DEMO_PLACE_SCENES: tuple[str, ...] = (
    "basketball_court",
    "sports_field",
    "street",
    "corridor_hallway",
    "house_interior",
    "compound_courtyard",
)


def _norm(name: str) -> str:
    return (
        (name or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def all_places() -> tuple[PlaceEntry, ...]:
    return PLACES


def places_by_group() -> dict[str, tuple[PlaceEntry, ...]]:
    groups: dict[str, list[PlaceEntry]] = {}
    for entry in PLACES:
        groups.setdefault(entry.group, []).append(entry)
    return {key: tuple(vals) for key, vals in groups.items()}


def place_alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in PLACES:
        mapping[_norm(entry.id)] = entry.id
        for alias in entry.aliases:
            mapping[_norm(alias)] = entry.id
    return mapping


_ALIAS_MAP = place_alias_map()
_BY_ID = {entry.id: entry for entry in PLACES}


def get_place(place_id: str) -> Optional[PlaceEntry]:
    return _BY_ID.get(_norm(place_id))


def resolve_place(name: str) -> Optional[PlaceEntry]:
    """Resolve a folder token, alias, or catalog id. Never a specific arena name."""
    key = _norm(name)
    if not key:
        return None
    place_id = _ALIAS_MAP.get(key)
    if place_id is None:
        return None
    return _BY_ID[place_id]


def place_display_name(place_id: str) -> str:
    entry = get_place(place_id)
    if entry is None:
        return (place_id or "").replace("_", " ").strip() or "unknown"
    return entry.display_name


def validate_place_type(place_id: str) -> str:
    """Return a canonical id or '' if unset. Reject unknown tokens."""
    cleaned = (place_id or "").strip()
    if not cleaned:
        return ""
    entry = resolve_place(cleaned)
    if entry is None:
        raise ValueError(
            f"Unknown place type '{place_id}'. Use a catalog id from vision/scene_context.py."
        )
    return entry.id


def default_place_for_sport(sport_id: str) -> Optional[str]:
    sport = resolve_sport(sport_id)
    key = sport.id if sport is not None else _norm(sport_id)
    return SPORT_DEFAULT_PLACE.get(key)


def default_category_for_place(place_id: str) -> str:
    entry = resolve_place(place_id)
    if entry is not None and entry.group == "sports":
        return "game_or_play"
    return "ordinary"


def is_sports_venue(place_id: Optional[str]) -> bool:
    return bool(place_id) and _norm(place_id) in SPORTS_VENUE_IDS


def is_confrontation_setting(place_id: Optional[str]) -> bool:
    return bool(place_id) and _norm(place_id) in CONFRONTATION_SETTING_IDS


def is_strong_confrontation_setting(place_id: Optional[str]) -> bool:
    return bool(place_id) and _norm(place_id) in STRONG_CONFRONTATION_IDS


def is_street_place(place_id: Optional[str]) -> bool:
    return _norm(place_id or "") == "street"


@dataclass
class KitCues:
    """Soft clothing-color proxies. Not identity, demographics, or affiliation."""

    team_kit_similarity: float = 0.0
    jersey_like_colors: bool = False
    note: str = (
        "Clothing color clusters are assistive only — not identity, "
        "demographics, or a guilt / gang label."
    )

    def to_dict(self) -> dict:
        return {
            "team_kit_similarity": round(float(self.team_kit_similarity), 3),
            "jersey_like_colors": bool(self.jersey_like_colors),
            "note": self.note,
        }


@dataclass
class PlaceAssessment:
    place_type: str = "unknown"
    confidence: float = 0.0
    display: str = "Unknown"
    source: str = "none"
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "place_type": self.place_type,
            "place_display": self.display or place_display_name(self.place_type),
            "place_confidence": round(float(self.confidence), 3),
            "place_source": self.source,
            "note": self.note,
        }


def analyze_kit_cues(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
) -> KitCues:
    """Compare saturated clothing colors across people.

    Similar hue clusters across 2+ people are a *team kit* proxy.
    Saturated blocks are a *jersey-like* proxy. Absence proves nothing.
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return KitCues()
    h, w = image_bgr.shape[:2]
    people = [d for d in detections if d.label.lower() == "person"]
    stats: list[tuple[float, float, float]] = []
    jersey = False
    for det in people:
        crop = _safe_crop(image_bgr, det.bbox_xyxy, h, w)
        if crop is None:
            continue
        hue, sat, frac = _dominant_saturated(crop)
        if sat >= 90 and frac >= 0.12:
            jersey = True
        if sat >= 40 and frac >= 0.08:
            stats.append((hue, sat, frac))
    similarity = 0.0
    if len(stats) >= 2:
        pairs = 0
        hits = 0
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                pairs += 1
                d_hue = _hue_dist(stats[i][0], stats[j][0])
                if d_hue <= 18 and min(stats[i][1], stats[j][1]) >= 50:
                    hits += 1
        similarity = hits / pairs if pairs else 0.0
    elif not people:
        hue, sat, frac = _dominant_saturated(image_bgr)
        jersey = sat >= 110 and frac >= 0.18
    return KitCues(
        team_kit_similarity=round(float(similarity), 3),
        jersey_like_colors=bool(jersey),
    )


def _safe_crop(
    image_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    h: int,
    w: int,
) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    xa, xb = int(max(0, min(w, x1))), int(max(0, min(w, x2)))
    ya, yb = int(max(0, min(h, y1))), int(max(0, min(h, y2)))
    if xb - xa < 4 or yb - ya < 4:
        return None
    # Torso-ish band: skip heads / feet.
    band_top = ya + (yb - ya) // 5
    band_bot = ya + 4 * (yb - ya) // 5
    crop = image_bgr[band_top:band_bot, xa:xb]
    if crop.size == 0:
        return None
    return crop


def _dominant_saturated(bgr: np.ndarray) -> tuple[float, float, float]:
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    small = cv2.resize(bgr, (32, 40), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    mask = (sat >= 60) & (val >= 50)
    frac = float(mask.mean())
    if frac < 0.04:
        return 0.0, float(sat.mean()), frac
    hues = hsv[:, :, 0][mask].astype(np.float32)
    return float(hues.mean()), float(sat[mask].mean()), frac


def _hue_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 180.0 - d)


def infer_place_heuristic(image_bgr: np.ndarray) -> tuple[Optional[str], float]:
    """Synthetic color/geometry proxies for demo painters.

    Honest: these are not a claim that live CCTV can be mapped onto every
    arena, street, or home. Scores below ~0.50 stay unknown.
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return None, 0.0
    if image_bgr.ndim < 2:
        return None, 0.0

    small = cv2.resize(image_bgr, (80, 60), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(small)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    mean_b = float(b.mean())
    mean_g = float(g.mean())
    mean_r = float(r.mean())
    sat_mean = float(sat.mean()) / 255.0
    if max(mean_b, mean_g, mean_r) < 16:
        return None, 0.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 130)
    edge_frac = float((edges > 0).mean())
    if edge_frac < 0.025:
        # Uniform color wash — do not invent a court, street, or home.
        return None, 0.0
    v_proj = (edges > 0).mean(axis=0)
    h_proj = (edges > 0).mean(axis=1)
    verticals = float((v_proj > 0.12).mean()) if v_proj.size else 0.0
    left_dark = float(val[:, :16].mean()) / 255.0
    right_dark = float(val[:, -16:].mean()) / 255.0
    center_bright = float(val[:, 28:52].mean()) / 255.0 if val.shape[1] >= 52 else 0.0
    side_dark = (left_dark + right_dark) / 2.0
    warmth = (mean_r - mean_b) / 255.0
    grayness = 1.0 - min(1.0, (abs(mean_r - mean_g) + abs(mean_g - mean_b) + abs(mean_r - mean_b)) / 180.0)

    orange = (
        (r.astype(np.int16) > 150)
        & (g.astype(np.int16) > 50)
        & (g.astype(np.int16) < 150)
        & (b.astype(np.int16) < 70)
    )
    green = (g > r) & (g > b) & (g > 80)
    blue = (b > r) & (b > g) & (b > 90)
    tan = (
        (r > 140)
        & (g > 110)
        & (b > 70)
        & (b < 160)
        & (r > b + 15)
        & (g > b)
    )
    orange_frac = float(orange.mean())
    green_frac = float(green.mean())
    blue_frac = float(blue.mean())
    tan_frac = float(tan.mean())

    scores = {
        "street": (
            0.10
            + 0.45 * grayness
            + 0.30 * verticals
            + (0.12 if 70 <= mean_r <= 150 and sat_mean < 0.28 else 0.0)
            - 0.45 * max(0.0, center_bright - side_dark)
            - 0.25 * orange_frac
            - 0.20 * green_frac
        ),
        "corridor_hallway": (
            0.08
            + 0.55 * max(0.0, center_bright - side_dark)
            + 0.15 * edge_frac
            + (0.20 if side_dark < 0.28 and center_bright > 0.50 else 0.0)
            - 0.20 * orange_frac
            - 0.15 * green_frac
        ),
        "house_interior": (
            0.08
            + 0.50 * max(0.0, warmth)
            + 0.15 * max(0.0, 0.22 - edge_frac)
            + (0.18 if mean_r > 150 and mean_b < 90 and sat_mean > 0.25 else 0.0)
            - 0.25 * verticals
        ),
        "compound_courtyard": (
            0.08
            + 0.50 * tan_frac
            + 0.12 * max(0.0, warmth)
            + (0.12 if tan_frac > 0.22 and edge_frac > 0.04 else 0.0)
            - 0.20 * orange_frac
        ),
        "basketball_court": 0.10 + 0.75 * orange_frac,
        "sports_field": 0.10 + 0.70 * green_frac - 0.20 * orange_frac,
        "tennis_court": 0.10 + 0.70 * blue_frac + 0.08 * sat_mean,
        "volleyball_court": 0.08 + 0.45 * tan_frac + 0.20 * float(h_proj[28:33].mean() if h_proj.size >= 33 else 0.0),
        "parking_lot": (
            0.06
            + 0.30 * grayness
            + 0.25 * float((h_proj > 0.10).mean() if h_proj.size else 0.0)
            - 0.20 * orange_frac
        ),
    }
    # Distinctive demo painters win their lane.
    vanishing = max(0.0, center_bright - side_dark)
    if grayness > 0.55 and verticals > 0.18 and sat_mean < 0.30 and orange_frac < 0.12 and vanishing < 0.22:
        scores["street"] = max(scores["street"], 0.72)
    if side_dark < 0.30 and center_bright > 0.50 and warmth < 0.25:
        scores["corridor_hallway"] = max(scores["corridor_hallway"], 0.74)
        scores["street"] = min(scores["street"], 0.40)
    if mean_r > 160 and mean_b < 85 and warmth > 0.25 and orange_frac < 0.35 and tan_frac < 0.16:
        scores["house_interior"] = max(scores["house_interior"], 0.70)
    if tan_frac > 0.24 and orange_frac < 0.20 and mean_r > 145:
        scores["compound_courtyard"] = max(scores["compound_courtyard"], 0.74)
        scores["house_interior"] = min(scores["house_interior"], 0.42)
    if orange_frac > 0.30 and mean_r > 150 and mean_b < 85:
        scores["basketball_court"] = max(scores["basketball_court"], 0.74)
        scores["house_interior"] = min(scores["house_interior"], 0.40)
    if green_frac > 0.35 and mean_g >= 120:
        scores["sports_field"] = max(scores["sports_field"], 0.72)

    place_id, raw = max(scores.items(), key=lambda kv: kv[1])
    conf = float(max(0.0, min(0.90, raw)))
    if conf < 0.55:
        return None, round(conf, 3)
    return place_id, round(conf, 3)


def place_from_folder_name(name: str) -> Optional[str]:
    """Extract a catalog place id from ``scene__street`` or a sport folder."""
    key = _norm(name)
    if not key:
        return None
    tokens = key.split("__")
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "scene" and i + 1 < len(tokens):
            entry = resolve_place(tokens[i + 1])
            return entry.id if entry is not None else None
        if tok.startswith("scene_"):
            entry = resolve_place(tok[6:])
            if entry is not None:
                return entry.id
        i += 1
    for tok in reversed(tokens):
        sport = resolve_sport(tok)
        if sport is not None:
            return default_place_for_sport(sport.id)
    return None


def _color_signature(image_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(image_bgr, (32, 24), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    hue_hist = hue_hist / (hue_hist.sum() + 1e-6)
    means = small.reshape(-1, 3).mean(axis=0) / 255.0
    sat = float(hsv[:, :, 1].mean()) / 255.0
    return np.concatenate([means, hue_hist, [sat]]).astype(np.float32)


@dataclass
class SceneFolderIndex:
    """Mean color signatures from labeled ``scene__*`` / sport folders."""

    prototypes: dict[str, np.ndarray] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    root: str = ""

    def infer(self, image_bgr: np.ndarray) -> tuple[Optional[str], float]:
        if not self.prototypes or image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return None, 0.0
        sig = _color_signature(image_bgr)
        best_id = None
        best_dist = 1e9
        for place_id, proto in self.prototypes.items():
            dist = float(np.linalg.norm(sig - proto))
            if dist < best_dist:
                best_dist = dist
                best_id = place_id
        # Typical same-class distance is ~0.15–0.45; far classes > 0.8.
        conf = float(max(0.0, min(0.88, 1.0 - best_dist / 1.35)))
        if conf < 0.58 or best_id is None:
            return None, round(conf, 3)
        return best_id, round(conf, 3)


_INDEX_CACHE: dict[str, SceneFolderIndex] = {}


def load_scene_folder_index(data_root: str | Path | None) -> SceneFolderIndex:
    """Build (and cache) prototypes from labeled activity folders."""
    if not data_root:
        return SceneFolderIndex()
    root = Path(data_root)
    cache_key = str(root.resolve()) if root.exists() else str(root)
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index = SceneFolderIndex(root=cache_key)
    if not root.is_dir():
        _INDEX_CACHE[cache_key] = index
        return index
    buckets: dict[str, list[np.ndarray]] = {}
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for child in split_dir.iterdir():
            if not child.is_dir():
                continue
            _collect_folder_signatures(child, buckets)
    for place_id, sigs in buckets.items():
        if not sigs:
            continue
        index.prototypes[place_id] = np.mean(np.stack(sigs, axis=0), axis=0)
        index.counts[place_id] = len(sigs)
    _INDEX_CACHE[cache_key] = index
    if index.prototypes:
        logger.info(
            "Scene folder index: %s place type(s) from %s",
            len(index.prototypes),
            root,
        )
    return index


def clear_scene_folder_index_cache() -> None:
    _INDEX_CACHE.clear()


def _collect_folder_signatures(
    folder: Path, buckets: dict[str, list[np.ndarray]], depth: int = 0
) -> None:
    place = place_from_folder_name(folder.name)
    if place:
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                buckets.setdefault(place, []).append(_color_signature(image))
            elif path.is_dir() and depth < 2:
                _collect_folder_signatures(path, buckets, depth + 1)
        return
    if depth < 2:
        for path in folder.iterdir():
            if path.is_dir():
                _collect_folder_signatures(path, buckets, depth + 1)


OBJECT_PRIOR_CONFIDENCE = 0.46
OBJECT_PRIOR_BUMP = 0.06


def place_prior_from_objects(object_ids: Optional[Sequence[str]]) -> Optional[PlaceAssessment]:
    """Weak place lean from catalog object presence.

    Fridge/sink → house_interior; bench/gate → outdoor/community. Confidence
    stays below the 0.50 policy threshold so this cannot flip fight-vs-play
    on its own. Camera stamps and labeled folders still win.
    """
    from vision.objects_catalog import OBJECT_PLACE_PRIOR, map_detector_label

    votes: dict[str, int] = {}
    for raw in object_ids or []:
        entry = map_detector_label(str(raw))
        oid = entry.id if entry is not None else str(raw).strip().lower()
        place = OBJECT_PLACE_PRIOR.get(oid)
        if not place:
            continue
        votes[place] = votes.get(place, 0) + 1
    if not votes:
        return None
    place_id = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return PlaceAssessment(
        place_type=place_id,
        confidence=OBJECT_PRIOR_CONFIDENCE,
        display=place_display_name(place_id),
        source="objects",
        note=(
            "Weak prior from detected home/community objects — not proof of "
            "the setting and not a named address. Humans verify. "
            "Place type is a catalog setting — not a named arena or address, "
            "and not a determination of what happened."
        ),
    )


def _apply_object_place_prior(
    assessment: PlaceAssessment,
    object_ids: Optional[Sequence[str]],
) -> PlaceAssessment:
    prior = place_prior_from_objects(object_ids)
    if prior is None:
        return assessment
    if assessment.place_type in {"", "unknown"} or assessment.source == "none":
        return prior
    if assessment.place_type == prior.place_type:
        bumped = min(0.70, float(assessment.confidence) + OBJECT_PRIOR_BUMP)
        return PlaceAssessment(
            place_type=assessment.place_type,
            confidence=bumped,
            display=assessment.display or place_display_name(assessment.place_type),
            source=assessment.source,
            note=(assessment.note or "")
            + " Object inventory gave a small same-place boost.",
        )
    # Conflicting objects never override camera / folder / heuristic.
    return assessment


def infer_scene_place(
    image_bgr: np.ndarray,
    *,
    camera_place_type: Optional[str] = None,
    data_root: str | Path | None = None,
    allow_heuristic: bool = True,
    object_ids: Optional[Sequence[str]] = None,
) -> PlaceAssessment:
    """Resolve place type.

    When the operator set ``camera.place_type``, that stamp wins (they know
    the feed). Otherwise labeled ``scene__*`` / sport folders are tried, then
    synthetic OpenCV proxies. Detected catalog objects may add a **small**
    home vs outdoor lean when place is still unknown. Never invents a
    specific arena name.
    """
    note_generic = (
        "Place type is a catalog setting — not a named arena or address, "
        "and not a determination of what happened."
    )
    stamped = (camera_place_type or "").strip()
    if stamped:
        try:
            place_id = validate_place_type(stamped)
        except ValueError:
            place_id = ""
        if place_id:
            return PlaceAssessment(
                place_type=place_id,
                confidence=0.95,
                display=place_display_name(place_id),
                source="camera",
                note="Operator-set camera place_type. " + note_generic,
            )

    if data_root:
        folder_id, folder_conf = load_scene_folder_index(data_root).infer(image_bgr)
        if folder_id and folder_conf >= 0.58:
            return PlaceAssessment(
                place_type=folder_id,
                confidence=folder_conf,
                display=place_display_name(folder_id),
                source="folder",
                note="Matched labeled scene / sport training folders. " + note_generic,
            )

    heuristic = PlaceAssessment(
        place_type="unknown",
        confidence=0.0,
        display="Unknown",
        source="none",
        note="No camera stamp, folder match, or distinctive heuristic. " + note_generic,
    )
    if allow_heuristic:
        hid, hconf = infer_place_heuristic(image_bgr)
        if hid and hconf >= 0.55:
            heuristic = PlaceAssessment(
                place_type=hid,
                confidence=hconf,
                display=place_display_name(hid),
                source="heuristic",
                note=(
                    "Synthetic color/geometry proxy for demo/MOCK painters — "
                    "not recognition of a real sports arena or street. "
                    + note_generic
                ),
            )

    return _apply_object_place_prior(heuristic, object_ids)


def kit_sport_confidence_boost(kit: KitCues, sport_confidence: float) -> float:
    """Slight play-confidence bump when kit colors cluster. Absence is a no-op."""
    if kit.team_kit_similarity < 0.55:
        return sport_confidence
    bump = 0.06 * float(kit.team_kit_similarity)
    return float(min(0.95, sport_confidence + bump))


def famous_arena_name_rejected(text: str) -> bool:
    """Guard used in tests: we never emit a specific arena as place_type."""
    lowered = (text or "").lower()
    banned = (
        "madison square garden",
        "wembley",
        "old trafford",
        "yankee stadium",
        "camp nou",
    )
    return any(name in lowered for name in banned)
