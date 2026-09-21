"""Catalog of common authorized-site sports and games.

This is a **structured lookup**, not a claim that the OpenCV + sklearn
activity model “knows most sports.” Operators can label folders so
``game_or_play`` carries an optional ``sport_context``. Runtime inference
is a best-effort color/geometry proxy aimed at synthetic demo scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Informal activity-folder names that mean "game" but not a specific sport.
GENERIC_GAME_TOKENS = frozenset(
    {
        "game_or_play",
        "game",
        "play",
        "sports",
        "sport",
        "roughhousing",
        "playful",
        "recreation",
    }
)


@dataclass(frozen=True)
class SportEntry:
    id: str
    display_name: str
    aliases: tuple[str, ...] = ()
    notes: str = ""


# Stable ids for authorized-site sports / games. Aim: most known, not infinite.
SPORTS: tuple[SportEntry, ...] = (
    SportEntry("basketball", "Basketball", ("hoops", "bball", "basket_ball")),
    SportEntry(
        "soccer",
        "Soccer / football",
        ("football", "association_football", "futbol", "football_soccer"),
        notes="International football. American football is a separate id.",
    ),
    SportEntry(
        "american_football",
        "American football",
        ("gridiron", "nfl_football", "americanfootball"),
    ),
    SportEntry("volleyball", "Volleyball", ("volley_ball", "volley")),
    SportEntry("tennis", "Tennis", ()),
    SportEntry("badminton", "Badminton", ()),
    SportEntry("baseball", "Baseball", ()),
    SportEntry("softball", "Softball", ()),
    SportEntry("cricket", "Cricket", ()),
    SportEntry("rugby", "Rugby", ("rugby_union", "rugby_league")),
    SportEntry(
        "hockey",
        "Hockey",
        ("ice_hockey", "field_hockey", "floor_hockey", "street_hockey"),
    ),
    SportEntry(
        "track_athletics",
        "Track / athletics",
        ("track", "athletics", "track_and_field", "running"),
    ),
    SportEntry(
        "wrestling",
        "Wrestling (sport)",
        ("wrestling_sport", "folkstyle", "freestyle_wrestling", "scholastic_wrestling"),
        notes="Sport wrestling, not a street confrontation.",
    ),
    SportEntry(
        "boxing",
        "Boxing (sport)",
        ("boxing_sport", "sport_boxing"),
        notes="Authorized sparring / sport boxing, not an assault label.",
    ),
    SportEntry(
        "martial_arts_training",
        "Martial arts training",
        ("martial_arts", "karate", "taekwondo", "judo", "bjj", "kung_fu"),
    ),
    SportEntry(
        "playground_games",
        "Playground games",
        ("playground", "recess_games", "tag", "hopscotch", "four_square"),
    ),
    SportEntry("table_tennis", "Table tennis", ("ping_pong", "pingpong")),
    SportEntry("gymnastics", "Gymnastics", ("artistic_gymnastics",)),
    SportEntry("swimming", "Swimming", ("swim", "pool_lap")),
    SportEntry("skateboarding", "Skateboarding", ("skateboard", "skate")),
    SportEntry("lacrosse", "Lacrosse", ("lax",)),
    SportEntry("golf", "Golf", ()),
    SportEntry("cycling", "Cycling", ("bike", "biking", "bicycle")),
    SportEntry("pickleball", "Pickleball", ()),
    SportEntry("ultimate_frisbee", "Ultimate frisbee", ("ultimate", "frisbee")),
    SportEntry("water_polo", "Water polo", ()),
    SportEntry("rowing", "Rowing", ("crew",)),
    SportEntry("fencing", "Fencing (sport)", ("sport_fencing",)),
    SportEntry("climbing", "Climbing", ("bouldering", "rock_climbing")),
    SportEntry("dodgeball", "Dodgeball", ("dodge_ball",)),
    SportEntry("kickball", "Kickball", ()),
    SportEntry("handball", "Handball", ("team_handball",)),
    SportEntry("squash", "Squash", ()),
    SportEntry("netball", "Netball", ()),
    SportEntry("cheer", "Cheer / sideline", ("cheerleading", "sideline_cheer")),
)

# Distinct synthetic painters used by Last run / activity demo frames.
DEMO_SPORT_SCENES: tuple[str, ...] = (
    "basketball",
    "soccer",
    "tennis",
    "volleyball",
    "american_football",
)
# Extra train-folder sports. Skip orange basketball so the tiny demo
# forest does not confuse it with the reddish fight painter.
DEMO_DATASET_SPORTS: tuple[str, ...] = (
    "soccer",
    "tennis",
    "volleyball",
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


def all_sports() -> tuple[SportEntry, ...]:
    return SPORTS


def sport_alias_map() -> dict[str, str]:
    """Map every catalog id and alias onto a canonical sport id."""
    mapping: dict[str, str] = {}
    for entry in SPORTS:
        mapping[_norm(entry.id)] = entry.id
        for alias in entry.aliases:
            mapping[_norm(alias)] = entry.id
    return mapping


_ALIAS_MAP = sport_alias_map()
_BY_ID = {entry.id: entry for entry in SPORTS}


def get_sport(sport_id: str) -> Optional[SportEntry]:
    return _BY_ID.get(_norm(sport_id))


def resolve_sport(name: str) -> Optional[SportEntry]:
    """Resolve a folder token, alias, or display-ish name to a catalog entry."""
    key = _norm(name)
    if not key or key in GENERIC_GAME_TOKENS:
        return None
    sport_id = _ALIAS_MAP.get(key)
    if sport_id is None:
        return None
    return _BY_ID[sport_id]


def sport_display_name(sport_id: str) -> str:
    entry = get_sport(sport_id)
    if entry is None:
        return (sport_id or "").replace("_", " ").strip() or "unknown"
    return entry.display_name


def infer_sport_context(
    image_bgr: np.ndarray,
) -> tuple[Optional[str], float]:
    """Best-effort color/geometry proxy → (sport_id, confidence).

    Tuned for the synthetic court painters. Real CCTV will usually score
    below the decent-confidence threshold — that is intentional and honest.
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
    mean_b = float(b.mean())
    mean_g = float(g.mean())
    mean_r = float(r.mean())
    if max(mean_b, mean_g, mean_r) < 18:
        return None, 0.0

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
        & (b < 150)
        & (r > b + 20)
        & (g > b)
    )
    brown = (r > 60) & (g > 35) & (b < 70) & (r > g) & (g > b)

    orange_frac = float(orange.mean())
    green_frac = float(green.mean())
    blue_frac = float(blue.mean())
    tan_frac = float(tan.mean())
    brown_frac = float(brown.mean())
    sat_mean = float(sat.mean()) / 255.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    edge_frac = float((edges > 0).mean())
    h_proj = (edges > 0).mean(axis=1)
    mid_band = float(h_proj[28:33].mean()) if h_proj.size >= 33 else 0.0
    v_proj = (edges > 0).mean(axis=0)
    yard_like = float((v_proj > 0.08).mean()) if v_proj.size else 0.0

    scores = {
        "basketball": 0.12
        + 0.78 * orange_frac
        + 0.12 * max(0.0, (mean_r - mean_b) / 255.0)
        + (0.06 if edge_frac > 0.04 else 0.0)
        - 0.40 * min(1.0, mid_band * 8.0),
        "soccer": 0.10
        + 0.75 * green_frac
        + 0.10 * max(0.0, (mean_g - mean_r) / 255.0)
        - 0.25 * orange_frac,
        "tennis": 0.12 + 0.78 * blue_frac + 0.08 * sat_mean,
        "volleyball": 0.10
        + 0.55 * tan_frac
        + 0.45 * min(1.0, mid_band * 10.0)
        - 0.35 * orange_frac,
        "american_football": 0.08
        + 0.20 * green_frac
        + 0.45 * brown_frac
        + 0.15 * yard_like
        + (0.06 if edge_frac > 0.06 else 0.0),
    }
    # Bright green pitch (demo soccer) vs darker olive + brown hash (football).
    if mean_g >= 125 and green_frac > 0.35 and orange_frac < 0.12:
        scores["soccer"] = max(scores["soccer"], 0.72)
        scores["american_football"] = min(scores["american_football"], 0.42)
    if mean_g < 115 and mean_r < 95 and (brown_frac > 0.06 or yard_like > 0.25):
        scores["american_football"] = max(scores["american_football"], 0.70)
        scores["soccer"] = min(scores["soccer"], 0.42)
    # Orange hardwood (demo basketball) vs tan gym with a net (volleyball).
    if mean_r > 150 and mean_b < 85 and mean_g < 140 and orange_frac > 0.30:
        scores["basketball"] = max(scores["basketball"], 0.74)
    if mid_band >= 0.10 and tan_frac > 0.20 and orange_frac < 0.25:
        scores["volleyball"] = max(scores["volleyball"], 0.68 + 0.2 * tan_frac)
        scores["basketball"] *= 0.55

    sport_id, raw = max(scores.items(), key=lambda kv: kv[1])
    conf = float(max(0.0, min(0.92, raw)))
    if conf < 0.50:
        return None, round(conf, 3)
    return sport_id, round(conf, 3)
