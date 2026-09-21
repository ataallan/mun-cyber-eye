# Scene context assist

**AI detects and alerts. Humans verify and decide.**

Operators want setting to help tell **game / play** from a **real confrontation**: a sports ground, a street, a corridor, someone’s compound, a house, an indoor arena. This module is a **structured scene-context assist** with explicit limits. It is **not** a claim that today’s OpenCV + sklearn stack can understand all sports arenas or read a scene the way a person does.

## Honesty

| Capability | What exists today | What it is not |
|------------|-------------------|----------------|
| Place / venue type | A catalog of ~26 generic settings (`basketball_court`, `street`, `corridor_hallway`, `house_interior`, `compound_courtyard`, …) | Recognition of a named arena (never “Madison Square Garden”), address, or private property owner |
| How place is chosen | 1) operator `place_type` on a camera 2) labeled `scene__*` / sport folders 3) synthetic OpenCV color/geometry proxies | Full scene understanding or a map of every court on earth |
| Uniform / kit | Similar clothing-color clusters across people; saturated jersey-like blocks | Who the person is, demographics, team identity, or “gang” / guilt labeling |
| Risk use | Setting + `sport_context` + body-aggression together | Proof of play, assault, or trespass |

Setting is **not** identity. A jersey-like color is **not** guilt.

## Place catalog

Stable ids live in `vision/scene_context.py`. Display names are generic.

**Sports / play venues:** `sports_field`, `basketball_court`, `tennis_court`, `volleyball_court`, `indoor_arena`, `gymnasium`, `playground`, `track`, `swimming_pool`, `skate_park`

**Circulation / public:** `street`, `sidewalk`, `parking_lot`, `corridor_hallway`, `lobby`, `stairwell`

**Residential / private:** `house_interior`, `residential_yard`, `compound_courtyard`, `driveway`

**Other:** `classroom_or_office`, `cafeteria`, `warehouse_or_industrial`, `parking_garage`, `outdoor_plaza`, `unknown`

Aliases (`hallway` → `corridor_hallway`, `compound` → `compound_courtyard`, `home` → `house_interior`, `pitch` → `sports_field`) resolve through the catalog. The pipeline never invents a specific arena or venue brand.

## How place is inferred

Priority:

1. **Camera registry** — when an operator sets `place_type` (and a free-text `location_label` such as “North corridor”), the pipeline **stamps** that catalog id. This is the authoritative feed label.
2. **Labeled training folders** — `scene__basketball_court/`, `game_or_play__scene__street/`, or a catalog sport folder that implies a default place (`basketball` → `basketball_court`). A lightweight color-signature index matches new frames to those folders when present.
3. **Heuristic OpenCV proxies** (demo / MOCK) — court-color painters, street-like gray verticals, corridor vanishing-line proxy, warm indoor wash, tan courtyard. These are **synthetic**. Real CCTV will often stay `unknown`. That is intentional.

`source` on the assessment is `camera` | `folder` | `heuristic` | `none`.

## Folder convention

```
data/activity/train/scene__street/*.jpg
data/activity/train/scene__basketball_court/*.jpg
data/activity/train/game_or_play__scene__street/*.jpg
data/activity/train/potential_fight__scene__corridor_hallway/*.jpg
data/activity/train/game_or_play__basketball/*.jpg   # implies basketball_court
```

Standalone `scene__<place>` folders map to `game_or_play` for sports venues and `ordinary` otherwise, unless a category prefix is present. Admin **Train models** can upload into a chosen place type or accept a zip that already uses these folder names.

`--generate-demo` writes a few `scene__*` folders so CI and Last run can exercise the assist. Those painters are not real places.

## Uniform / kit cues

`vision/scene_context.analyze_kit_cues` looks at person boxes:

- `team_kit_similarity` — fraction of person pairs with similar saturated hue (soft team-kit proxy)
- `jersey_like_colors` — saturated color blocks that look jersey-like

Absence of matching kits **does not** prove a fight. Matching kits **slightly** boost `sport_confidence` when a sport is already named. The assist never identifies a person and never attaches a criminal or affiliation label to clothes.

## Fight vs play (setting + sport + aggression)

Documented policy (tests cover the combinations):

1. **Strong sports venue** + `sport_context` → stronger soften toward `game_or_play` (even if sport confidence is only modest).
2. **Street / corridor / house / compound** + high body-aggression + **no** `sport_context` → stronger lean `potential_fight` (alert; risk may be `high`).
3. **Street** + `sport_context` (street soccer, …) → still `game_or_play` unless aggression is **extreme**; rationale notes *“street play — verify.”*
4. Kit similarity boosts play confidence slightly; missing kits do not create a fight.
5. Weapon-object and fall paths are unchanged.
6. Face aggression remains **off** by default and never identifies anyone.

See [SPORTS_AND_AGGRESSION.md](SPORTS_AND_AGGRESSION.md).

## Console

- **Cameras** — optional Place type dropdown + location/venue notes. Demo file camera is stamped `gymnasium`; the RTSP stub is `street`.
- **Run Pipeline → Last run** — `place_type`, kit cues, `sport_context`, and aggression together.
- **Alert detail** — scene place and kit metadata in the rationale / payload.
- **Admin train** — place dropdown and `scene__*` zip paths; inventory lists place-folder counts.

## Limits (again)

This is not production scene understanding, not a sports broadcaster, and not clothing-based identity or guilt. False positives and false negatives are expected. Humans verify every consequential call.
