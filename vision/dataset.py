"""Activity-recognition dataset layout and synthetic demo generation.

Expected tree (under ``data/activity`` by default)::

    data/activity/
      train/<category>/*.jpg
      val/<category>/*.jpg
      test/<category>/*.jpg

Canonical categories:

    ordinary
    game_or_play
    dance
    potential_fight
    potential_fall
    potential_weapon_object

Related folder names (fight, confrontation, altercation, play, dance, fall,
weapon, …) are accepted as aliases when loading. Sport / game folders map
onto ``game_or_play`` plus an optional ``sport_context``::

    train/game_or_play__basketball/*.jpg
    train/game_or_play/basketball/*.jpg
    train/basketball/*.jpg

``--generate-demo`` paints simple geometric scenes so CI and laptops can
train without real CCTV — it is **not** a substitute for authorized video.
Game vs fight is hard even for humans; these painters are honest synthetic
proxies, not real activity. Named sports are a catalog + color/geometry
assist, not a claim that the model knows most sports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from vision.features import extract_frame_features
from vision.sports_catalog import (
    DEMO_SPORT_SCENES,
    resolve_sport,
    sport_display_name,
)

logger = logging.getLogger(__name__)

ACTIVITY_CATEGORIES: Tuple[str, ...] = (
    "ordinary",
    "game_or_play",
    "dance",
    "potential_fight",
    "potential_fall",
    "potential_weapon_object",
)

# Related names from heuristics / informal labels → canonical category
CATEGORY_ALIASES = {
    "ordinary": "ordinary",
    "normal": "ordinary",
    "ok": "ordinary",
    "game_or_play": "game_or_play",
    "game": "game_or_play",
    "play": "game_or_play",
    "sports": "game_or_play",
    "sport": "game_or_play",
    "roughhousing": "game_or_play",
    "playful": "game_or_play",
    "recreation": "game_or_play",
    "dance": "dance",
    "dancing": "dance",
    "choreography": "dance",
    "choreographed": "dance",
    "potential_fight": "potential_fight",
    "fight": "potential_fight",
    "assault": "potential_fight",
    "confrontation": "potential_fight",
    "altercation": "potential_fight",
    "potential_fall": "potential_fall",
    "fall": "potential_fall",
    "person_down": "potential_fall",
    "collapse": "potential_fall",
    "potential_weapon_object": "potential_weapon_object",
    "weapon": "potential_weapon_object",
    "weapon_object": "potential_weapon_object",
    "potential_weapon": "potential_weapon_object",
}

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_folder_label(name: str) -> Tuple[str, Optional[str]]:
    """Map a folder name onto (canonical_category, optional sport_context).

    Accepts ``game_or_play__basketball``, a bare sport id/alias, and the
    existing activity aliases (``confrontation`` → ``potential_fight``).
    """
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    sport_ctx: Optional[str] = None
    cat_key = key
    if "__" in key:
        left, right = key.split("__", 1)
        sport = resolve_sport(right)
        if sport is not None:
            sport_ctx = sport.id
        cat_key = left

    sport = resolve_sport(cat_key)
    if sport is not None and cat_key not in CATEGORY_ALIASES:
        return "game_or_play", sport.id

    if cat_key not in CATEGORY_ALIASES:
        raise ValueError(
            f"Unknown activity category '{name}'. "
            f"Use one of: {', '.join(ACTIVITY_CATEGORIES)} "
            f"or a catalog sport (see vision/sports_catalog.py)."
        )
    return CATEGORY_ALIASES[cat_key], sport_ctx


def canonicalize_category(name: str) -> str:
    return parse_folder_label(name)[0]


def ensure_dataset_tree(root: str | Path) -> Path:
    """Create empty split/category folders (dataset layout under data/)."""
    root_path = Path(root)
    for split in SPLITS:
        for cat in ACTIVITY_CATEGORIES:
            (root_path / split / cat).mkdir(parents=True, exist_ok=True)
    return root_path


def _yield_labeled_images(
    folder: Path, category: str, sport: Optional[str]
) -> Iterator[Tuple[Path, str, Optional[str]]]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path, category, sport
        elif path.is_dir():
            try:
                nested_cat, nested_sport = parse_folder_label(path.name)
            except ValueError:
                logger.warning("Skipping unrecognized class folder: %s", path)
                continue
            use_sport = nested_sport or sport
            use_cat = nested_cat if nested_sport is None else category
            if nested_sport and category == "game_or_play":
                use_cat = "game_or_play"
            yield from _yield_labeled_images(path, use_cat, use_sport)


def iter_split_samples_with_context(
    root: str | Path, split: str
) -> Iterator[Tuple[Path, str, Optional[str]]]:
    """Yield (image_path, canonical_category, sport_context) for a split."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        return
    for child in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        try:
            category, sport = parse_folder_label(child.name)
        except ValueError:
            logger.warning("Skipping unrecognized class folder: %s", child)
            continue
        yield from _yield_labeled_images(child, category, sport)


def iter_split_samples(
    root: str | Path, split: str
) -> Iterator[Tuple[Path, str]]:
    """Yield (image_path, canonical_category) for a split."""
    for path, category, _sport in iter_split_samples_with_context(root, split):
        yield path, category


def load_split_features(
    root: str | Path,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load images in ``root/split`` and return (X, y, paths)."""
    xs: List[np.ndarray] = []
    ys: List[str] = []
    paths: List[str] = []
    for path, category in iter_split_samples(root, split):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Could not read %s", path)
            continue
        xs.append(extract_frame_features(image))
        ys.append(category)
        paths.append(str(path))
    if not xs:
        return (
            np.zeros((0, extract_frame_features(np.zeros((48, 64, 3), np.uint8)).size)),
            np.array([], dtype=object),
            [],
        )
    return np.stack(xs, axis=0), np.array(ys, dtype=object), paths


def render_demo_frame(
    category: str,
    seed: int = 0,
    width: int = 160,
    height: int = 120,
    sport_context: Optional[str] = None,
) -> np.ndarray:
    """Paint a class-typical synthetic scene (demo / tests only)."""
    category, parsed_sport = parse_folder_label(category)
    sport = sport_context or parsed_sport
    if sport:
        resolved = resolve_sport(sport)
        sport = resolved.id if resolved is not None else sport
    rng = np.random.default_rng(int(seed))
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if category == "game_or_play" and sport:
        _paint_sport(img, rng, sport)
    else:
        painters = {
            "ordinary": _paint_ordinary,
            "game_or_play": _paint_game,
            "dance": _paint_dance,
            "potential_fight": _paint_fight,
            "potential_fall": _paint_fall,
            "potential_weapon_object": _paint_weapon,
        }
        painters.get(category, _paint_ordinary)(img, rng)
    _add_noise(img, rng, sigma=6)
    return img


def generate_demo_dataset(
    root: str | Path,
    n_train: int = 40,
    n_val: int = 12,
    n_test: int = 12,
    seed: int = 7,
    overwrite: bool = False,
) -> Path:
    """Write synthetic JPEGs into the standard split/category layout."""
    root_path = ensure_dataset_tree(root)
    counts = {"train": n_train, "val": n_val, "test": n_test}
    offset = 0
    written = 0
    for split in SPLITS:
        n = counts[split]
        for c_i, cat in enumerate(ACTIVITY_CATEGORIES):
            dest = root_path / split / cat
            for i in range(n):
                path = dest / f"demo_{cat}_{i:04d}.jpg"
                if path.exists() and not overwrite:
                    continue
                frame = render_demo_frame(
                    cat, seed=seed + offset + c_i * 1000 + i
                )
                ok = cv2.imwrite(str(path), frame)
                if not ok:
                    raise RuntimeError(f"Failed to write {path}")
                written += 1
        offset += 10_000
        if n <= 0:
            continue
        n_sport = max(2, n // 3)
        for s_i, sport in enumerate(DEMO_SPORT_SCENES):
            dest = root_path / split / f"game_or_play__{sport}"
            dest.mkdir(parents=True, exist_ok=True)
            for i in range(n_sport):
                path = dest / f"demo_game_or_play__{sport}_{i:04d}.jpg"
                if path.exists() and not overwrite:
                    continue
                frame = render_demo_frame(
                    "game_or_play",
                    seed=seed + offset + 50_000 + s_i * 100 + i,
                    sport_context=sport,
                )
                ok = cv2.imwrite(str(path), frame)
                if not ok:
                    raise RuntimeError(f"Failed to write {path}")
                written += 1
    logger.info("Demo dataset at %s (wrote %s new images)", root_path, written)
    return root_path


def _add_noise(img: np.ndarray, rng: np.random.Generator, sigma: float) -> None:
    noise = rng.normal(0, sigma, img.shape)
    np.clip(img.astype(np.float32) + noise, 0, 255, out=noise)
    img[:] = noise.astype(np.uint8)


def _paint_game(img: np.ndarray, rng: np.random.Generator) -> None:
    """Generic sports / play: green field, spaced figures, a ball."""
    _paint_soccer(img, rng)


def _paint_sport(img: np.ndarray, rng: np.random.Generator, sport: str) -> None:
    painters = {
        "basketball": _paint_basketball,
        "soccer": _paint_soccer,
        "tennis": _paint_tennis,
        "volleyball": _paint_volleyball,
        "american_football": _paint_american_football,
    }
    painters.get(sport, _paint_soccer)(img, rng)


def _paint_soccer(img: np.ndarray, rng: np.random.Generator) -> None:
    """Green pitch, white center circle, white/black ball."""
    h, w = img.shape[:2]
    img[:] = (
        int(rng.integers(25, 55)),
        int(rng.integers(140, 190)),
        int(rng.integers(15, 45)),
    )
    cv2.line(img, (0, h // 2), (w - 1, h // 2), (220, 230, 240), 2)
    cv2.circle(img, (w // 2, h // 2), max(10, w // 7), (230, 235, 240), 1)
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (210, 220, 230), 1)
    _draw_person(img, int(w * 0.16), standing=True, color=(30, 90, 220), rng=rng)
    _draw_person(img, int(w * 0.70), standing=True, color=(20, 200, 240), rng=rng)
    cx = int(rng.integers(max(12, w // 2 - 12), min(w - 12, w // 2 + 12)))
    cy = int(rng.integers(int(h * 0.42), int(h * 0.68)))
    cv2.circle(img, (cx, cy), 8, (230, 230, 230), -1)
    cv2.circle(img, (cx, cy), 8, (20, 20, 20), 1)


def _paint_basketball(img: np.ndarray, rng: np.random.Generator) -> None:
    """Orange court, white key, orange ball."""
    h, w = img.shape[:2]
    img[:] = (
        int(rng.integers(20, 50)),
        int(rng.integers(70, 110)),
        int(rng.integers(175, 220)),
    )
    cv2.rectangle(img, (w // 3, 8), (2 * w // 3, h - 8), (235, 235, 240), 2)
    cv2.circle(img, (w // 2, int(h * 0.62)), max(8, w // 8), (235, 235, 240), 1)
    _draw_person(img, int(w * 0.18), standing=True, color=(40, 40, 40), rng=rng)
    _draw_person(img, int(w * 0.68), standing=True, color=(30, 30, 200), rng=rng)
    cx = int(rng.integers(int(w * 0.42), int(w * 0.58)))
    cy = int(rng.integers(int(h * 0.40), int(h * 0.70)))
    cv2.circle(img, (cx, cy), 8, (20, 120, 250), -1)


def _paint_tennis(img: np.ndarray, rng: np.random.Generator) -> None:
    """Blue hard court, white lines, yellow-green ball."""
    h, w = img.shape[:2]
    img[:] = (
        int(rng.integers(150, 200)),
        int(rng.integers(70, 110)),
        int(rng.integers(20, 50)),
    )
    cv2.rectangle(img, (10, 8), (w - 11, h - 9), (235, 235, 240), 1)
    cv2.line(img, (w // 2, 8), (w // 2, h - 9), (235, 235, 240), 1)
    cv2.line(img, (10, h // 2), (w - 11, h // 2), (235, 235, 240), 1)
    _draw_person(img, int(w * 0.20), standing=True, color=(20, 180, 240), rng=rng)
    _draw_person(img, int(w * 0.66), standing=True, color=(240, 240, 240), rng=rng)
    cx = int(rng.integers(int(w * 0.40), int(w * 0.60)))
    cy = int(rng.integers(int(h * 0.35), int(h * 0.60)))
    cv2.circle(img, (cx, cy), 6, (40, 230, 230), -1)


def _paint_volleyball(img: np.ndarray, rng: np.random.Generator) -> None:
    """Tan indoor court, white net, yellow/white ball."""
    h, w = img.shape[:2]
    img[:] = (
        int(rng.integers(70, 110)),
        int(rng.integers(130, 170)),
        int(rng.integers(170, 210)),
    )
    cv2.line(img, (0, h // 2), (w - 1, h // 2), (245, 245, 250), 3)
    cv2.line(img, (w // 2, h // 2 - 12), (w // 2, h // 2 + 12), (245, 245, 250), 2)
    _draw_person(img, int(w * 0.18), standing=True, color=(30, 80, 200), rng=rng)
    _draw_person(img, int(w * 0.64), standing=True, color=(40, 160, 40), rng=rng)
    cx = int(rng.integers(int(w * 0.40), int(w * 0.58)))
    cy = int(rng.integers(int(h * 0.22), int(h * 0.40)))
    cv2.circle(img, (cx, cy), 7, (40, 230, 240), -1)


def _paint_american_football(img: np.ndarray, rng: np.random.Generator) -> None:
    """Green field, yard lines, brown oval ball."""
    h, w = img.shape[:2]
    img[:] = (
        int(rng.integers(20, 45)),
        int(rng.integers(90, 130)),
        int(rng.integers(15, 40)),
    )
    for x in range(12, w - 8, max(14, w // 8)):
        cv2.line(img, (x, 6), (x, h - 7), (220, 225, 230), 1)
    _draw_person(img, int(w * 0.22), standing=True, color=(20, 20, 180), rng=rng)
    _draw_person(img, int(w * 0.62), standing=True, color=(20, 20, 40), rng=rng)
    cx = int(rng.integers(int(w * 0.40), int(w * 0.58)))
    cy = int(rng.integers(int(h * 0.45), int(h * 0.65)))
    cv2.ellipse(img, (cx, cy), (10, 6), 25, 0, 360, (20, 55, 90), -1)


def _paint_dance(img: np.ndarray, rng: np.random.Generator) -> None:
    """Dance: magenta stage, choreographed arcs — distinct from fight streaks."""
    h, w = img.shape[:2]
    img[:] = (
        int(rng.integers(140, 185)),
        int(rng.integers(20, 55)),
        int(rng.integers(130, 185)),
    )
    center = (w // 2, int(h * 0.62))
    for radius in (16, 26, 36):
        cv2.ellipse(
            img,
            center,
            (radius, max(6, radius // 2)),
            0,
            200,
            340,
            (240, 180, 240),
            2,
        )
    for i in range(8):
        ang = (i / 8.0) * 2 * np.pi
        px = int(np.clip(w * 0.5 + np.cos(ang) * w * 0.28, 4, w - 5))
        py = int(np.clip(h * 0.45 + np.sin(ang) * h * 0.22, 4, h - 5))
        cv2.circle(img, (px, py), 3, (250, 230, 250), -1)
    _draw_person(img, int(w * 0.26), standing=True, color=(220, 80, 220), rng=rng)
    _draw_person(img, int(w * 0.54), standing=True, color=(240, 140, 80), rng=rng)


def _paint_ordinary(img: np.ndarray, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    # Cool, calm interior
    img[:] = (
        int(rng.integers(90, 130)),  # B
        int(rng.integers(50, 80)),
        int(rng.integers(20, 50)),
    )
    x = int(rng.integers(w // 3, (2 * w) // 3 - 16))
    _draw_person(img, x, standing=True, color=(70, 80, 110), rng=rng)


def _paint_fight(img: np.ndarray, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    img[:] = (
        int(rng.integers(20, 50)),
        int(rng.integers(20, 50)),
        int(rng.integers(140, 200)),  # reddish
    )
    _draw_person(img, int(rng.integers(20, w // 3)), standing=True, color=(40, 40, 180), rng=rng)
    _draw_person(img, int(rng.integers(w // 3, w // 2)), standing=True, color=(50, 60, 200), rng=rng)
    # Motion / strike streaks
    for _ in range(int(rng.integers(4, 8))):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, (220, 220, 255), thickness=2)


def _paint_fall(img: np.ndarray, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    img[:] = (
        int(rng.integers(70, 100)),
        int(rng.integers(90, 120)),
        int(rng.integers(70, 100)),
    )
    x = int(rng.integers(15, max(16, w // 5)))
    y = int(rng.integers(int(h * 0.55), int(h * 0.72)))
    bw = int(rng.integers(int(w * 0.45), int(w * 0.7)))
    bh = int(rng.integers(14, 22))
    cv2.ellipse(
        img,
        (x + bw // 2, y + bh // 2),
        (bw // 2, bh // 2),
        0,
        0,
        360,
        (50, 70, 90),
        -1,
    )
    cv2.circle(img, (x + 12, y + bh // 2), 7, (45, 65, 85), -1)


def _paint_weapon(img: np.ndarray, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    img[:] = (
        int(rng.integers(40, 70)),
        int(rng.integers(70, 110)),
        int(rng.integers(20, 40)),
    )
    x = int(rng.integers(30, w // 2))
    _draw_person(img, x, standing=True, color=(40, 90, 70), rng=rng)
    # Bright yellow object near the "hand"
    ox = x + int(rng.integers(20, 36))
    oy = int(rng.integers(int(h * 0.35), int(h * 0.5)))
    cv2.rectangle(img, (ox, oy), (ox + 14, oy + 8), (10, 230, 240), -1)


def _draw_person(
    img: np.ndarray,
    x: int,
    standing: bool,
    color: tuple[int, int, int],
    rng: np.random.Generator,
) -> None:
    h, w = img.shape[:2]
    body_h = int(h * (0.55 if standing else 0.22))
    body_w = int(w * 0.12)
    top = int(h * 0.18) + int(rng.integers(-4, 5))
    x = int(np.clip(x, 2, w - body_w - 2))
    cv2.rectangle(img, (x, top), (x + body_w, min(h - 4, top + body_h)), color, -1)
    cv2.circle(img, (x + body_w // 2, max(8, top - 6)), 8, color, -1)


def count_split(root: str | Path, split: str) -> dict[str, int]:
    counts = {cat: 0 for cat in ACTIVITY_CATEGORIES}
    for _, cat in iter_split_samples(root, split):
        counts[cat] += 1
    return counts


def describe_dataset(root: str | Path) -> dict:
    root_path = Path(root)
    sports: dict[str, int] = {}
    for split in SPLITS:
        for _, _cat, sport in iter_split_samples_with_context(root_path, split):
            if sport:
                sports[sport] = sports.get(sport, 0) + 1
    return {
        "root": str(root_path),
        "categories": list(ACTIVITY_CATEGORIES),
        "splits": {split: count_split(root_path, split) for split in SPLITS},
        "sport_contexts": {
            sport: {"id": sport, "label": sport_display_name(sport), "frames": n}
            for sport, n in sorted(sports.items())
        },
    }
