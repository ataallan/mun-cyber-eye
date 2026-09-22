# Dangerous objects and use-against-person intensity

**AI detects and alerts. Humans verify and decide.**

The home/community inventory (`vision/objects_catalog.py`) is furniture and civic structures. This catalog is a **separate** assist for objects that can hurt a person. It is not a complete weapons encyclopedia, not ballistic proof, and not a guilt label. Toys, tools, and phones false-fire.

## Catalog

Stable ids live in `vision/dangerous_objects.py`. Each row is marked `dangerous_object` with `harm_potential` `low` / `medium` / `high`.

| Class | Examples | Notes |
|-------|----------|--------|
| `edged` | knife, machete, scissors (proxy) | Knife has YOLO/COCO overlap |
| `blunt` | baseball_bat, crowbar, metal_pipe, heavy_stick | Bat may sport-soften at low intensity |
| `firearm_like` | handgun, rifle | Visual proxy only — **never** ballistic certainty |
| `improvised` | bottle, brick, chair_as_weapon | Context-dependent: a bottle on a table or an ordinary chair is **not** a weapon |
| `chemical_fire` | spray_can, fire_proxy | Optional, honest, weak. Not a chemical-weapon or arson classifier |

Developer **Extract frames from video** can target activity class `potential_weapon_object` or kind **dangerous / weapon-like class** (`data/dangerous/train/<id>/`).

## Use-against-person intensity (0–1)

CPU bbox / motion proxies when a catalog object (or `raised_object`) co-occurs with person(s):

| `use_intensity_label` | Typical score | Cues |
|-----------------------|---------------|------|
| `none` | ~0.0–0.25 | Dedicated object present; use not observed |
| `brandished` | ~0.48 | Raised / firearm-like in hand; one person = brandish only |
| `threatening_motion` | ~0.68 | Toward another person’s bbox, closing distance + aggression, or strike motion nearby |
| `possible_strike` | ~0.86 | Strike-like vertical motion **and** toward / closing / raised |

`firearm_aimed_at_person` remains a separate geometry cue (intensity ~0.92) and is **never** sport-softened.

Rationale always says this is an **indicator only** — not proof of assault or intent.

## Risk policy

- Dedicated weapon-like objects still take the high-priority `potential_weapon_object` path for human review.
- **Sport:** baseball bat on a baseball / softball / cricket field with **low** use-intensity → `game_or_play` / intense play.
- Bat + **high** use-intensity toward a person **outside** sport → weapon / confrontation alert.
- Improvised classes require use cues (raised, toward, strike, thrown). A sitting bottle or idle chair does not alert.
- An object that is **not** in the home catalog and **not** a named dangerous class can still raise `unidentified_striking_object` when it moves into contact with another person, or an unidentified thrown-object cue when it is thrown at a person. A dead stream or a still frame does not invent that cue. Label those frames as `unidentified_improvised` from the Objects gallery.
- Alert metadata includes `weapon_class`, `harm_potential`, `use_intensity`, `use_intensity_label`.

## Console

Last run lists dangerous-object ids and max use intensity. Alert detail shows class, harm potential, and intensity label. Scene-assist Safety cues include the catalog id when present.

## Limits (again)

False positives: toys, kitchen knives in a kitchen, phones, umbrellas, sports equipment, office scissors. False negatives: occluded or novel objects. Humans verify. The system does not enforce, detain, or dispatch.

See [ETHICS_AND_SAFETY.md](ETHICS_AND_SAFETY.md), [SPORTS_AND_AGGRESSION.md](SPORTS_AND_AGGRESSION.md), [GUNSHOTS_AND_FALLS.md](GUNSHOTS_AND_FALLS.md).
