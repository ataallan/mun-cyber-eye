# Gunshots, falls, aimed firearms, thrown objects

**AI detects and alerts. Humans verify and decide.**

These assists sit on top of the existing activity / risk pipeline. They are **weak CPU / bbox / frame-to-frame proxies**. They do not medically diagnose a collapse, prove a gunshot, prove a real firearm, or prove assault. The system does **not** dispatch, detain, or enforce.

## Fall manner (`potential_fall` stays the alert)

Person-down still queues a **potential_fall** alert. A subtype is attached when cues exist:

| `fall_manner` | Meaning | Honesty |
|---------------|---------|---------|
| `sudden_collapse` / `unprecedented_fall` | Standing-then-down, vertical drop, abrupt motion | Not syncope vs assault |
| `accidental_fall` | Wide / horizontal pose, milder motion, no weapon-like context | Not a confirmed trip |
| `unknown_fall` | Mixed or weak cues | Default when we cannot tell |

Vision cannot diagnose syncope, assault, or a trip. Reviewers still treat every person-down as needing a look.

## Gunshot video proxy (`potential_gunshot`)

Reliable gunshot detection is usually **acoustic**. This prototype is video-first.

- Cue: `possible_gunshot_video_proxy` when a **combination** of localized flash, multi-person dive-like motion, and/or a firearm-like object is present.
- Risk category: `potential_gunshot` (risk-engine enum — **not** a sklearn activity class).
- Rationale: *Possible gunshot video proxy — verify, not a confirmed gunshot.* Fireworks, reflections, and camera artifacts false-fire.
- Audio: `ENABLE_GUNSHOT_AUDIO=0` (default). The stub reports `disabled` / `unavailable` and **never invents a bang**, even if a wav file is present (no acoustic model is bundled).

## Aimed firearm-like object (`firearm_aimed_at_person`)

Geometry: long-axis tip of a firearm-like / aimable bbox from the nearest holder, cone toward **another** person’s bbox.

| Scene | Result |
|-------|--------|
| Two+ people + ray toward another person | `aimed_at_person` / use tier `aimed_at_person`. **High / critical.** Sport context must **not** suppress this. |
| One person + gun / knife | Brandish / presence only — not aimed-at-person. |
| Bat on a field | May still sport-soften as a dangerous object; **aimed firearm does not**. |

Rationale: *Possible firearm-like object pointed toward a person — verify. Not proof of a real firearm or intent.*

## Thrown object toward a person (`object_thrown_at_person`)

Use tier: `thrown_projectile`. Distinct from an aimed firearm and from a static brandish. Can co-exist with blunt / improvised dangerous objects.

CPU proxies (honest, weak):

- Small/medium object (or dangerous / improvised class) with **fast translation** across frames
- Motion vector generally **toward another person’s bbox** (closing distance)
- Optional: release-like motion from a holder arm region, then the object separates

Risk: **elevated / high** for human review. Rationale: *Possible object thrown toward a person — verify. Not proof of assault.*

Sport soften:

- Ball + `sport_context` + sports ball / low-harm object on a court or field → prefer `game_or_play` / intense play unless aggression **and** an aimed-at-person firearm cue is strong.
- Brick / bottle / improvised toward a person on a **street / corridor / house** → **do not** soften.

Priority in the risk engine: aimed firearm → gunshot proxy → thrown object (if not sport-softened) → activity class → weapon presence → fight → fall.

## Console

- **Run Pipeline → Last run** lists fall-manner counts, gunshot-proxy frames (with audio status), `firearm_aimed_at_person` frames, and `object_thrown_at_person` frames. The scene-assist table has a Safety cues column.
- **Alert detail** shows fall manner, gunshot proxy (separate from weapon intensity), aimed-firearm geometry, and thrown-object cue when present in metadata.

## Config

```bash
ENABLE_GUNSHOT_AUDIO=0   # default OFF — never invent audio detections
ENABLE_FACE_AGGRESSION=0
```

See [ETHICS_AND_SAFETY.md](ETHICS_AND_SAFETY.md), [SPORTS_AND_AGGRESSION.md](SPORTS_AND_AGGRESSION.md), [OBJECTS_AND_STRUCTURES.md](OBJECTS_AND_STRUCTURES.md).
