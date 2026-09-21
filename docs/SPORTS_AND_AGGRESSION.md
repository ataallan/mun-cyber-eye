# Sports context and aggression assists

**AI detects and alerts. Humans verify and decide.**

This layer helps operators tell **authorized play** from a **possible confrontation**. It is a structured assist — not a claim that today’s OpenCV + sklearn activity model “knows most sports,” and not a claim that it can reliably read facial aggression.

## Honesty

| Capability | What exists today | What it is not |
|------------|-------------------|----------------|
| Sports / games familiarity | A **catalog** of ~35 common authorized-site sports, folder labels, and color/geometry proxies on synthetic courts | Recognition of real leagues, kits, or “most games” on live CCTV |
| Body aggressiveness | CPU OpenCV motion / proximity / raised-arm **proxies** | Proof of assault, intent, or who started it |
| Face aggressiveness | **Off by default.** Optional Haar + geometric tension proxy for lab use | Identity, demographics, emotion recognition, or a criminal label |

Retrain the Phase 3 activity model on authorized labeled video before any pilot. Game vs fight is hard even for humans.

## Sports catalog

Stable ids live in `vision/sports_catalog.py` (basketball, soccer / football, American football, volleyball, tennis, badminton, baseball, softball, cricket, rugby, hockey, track / athletics, wrestling (sport), boxing (sport), martial arts training, playground games, table tennis, gymnastics, swimming, skateboarding, and others).

Operators can attach a `sport_context` to `game_or_play` with either folder convention:

```
data/activity/train/game_or_play__basketball/*.jpg
data/activity/train/game_or_play/basketball/*.jpg
data/activity/train/basketball/*.jpg
```

Aliases (`football` → soccer, `hoops` → basketball, `wrestling_sport` → wrestling, …) resolve through the catalog. Bare `sports` / `play` still mean generic `game_or_play` with **no** specific sport.

`--generate-demo` paints a few distinct courts (orange key, green pitch, blue tennis, tan volleyball, yard lines) so the demo path and MOCK can emit `sport_context` on Last run. Those painters are not real activity.

## Body-aggression cues

`vision/aggression.py` compares the current frame to the previous one:

- Motion intensity and sudden acceleration
- Close proximity + opposing / approaching motion (clash proxy)
- Raised-arm / striking-like vertical motion in the upper band
- Uniform whole-frame change is treated as a **scene cut**, not a fight

Soft labels (`aggressive_motion`, `aggressive_pose`, plus the existing `rapid_motion` / `strike_motion` / `close_proximity`) are wired into `RiskEngine.FIGHT_SIGNALS`.

## Fight vs sport policy

Documented rule (also covered by tests):

1. **High aggression + no `sport_context`** → lean `potential_fight` and **alert**.
2. **High aggression + decent/strong `sport_context`** → stay `game_or_play` with rationale *“possible intense play; human should verify.”* **No threat alert** unless `ALERT_ON_INTENSE_SPORT=1`.
3. **Low aggression + `sport_context`** → `game_or_play`, no threat alert.
4. Phase 2 fight heuristics (`close_proximity` + `rapid_motion`, …) are **softened** when `sport_context` is present with decent confidence and aggression is not high.

Weapon-object and fall paths are unchanged. Dance stays dance (choreography is not upgraded to a fight on scene-cut motion).

```bash
# .env
ALERT_ON_INTENSE_SPORT=0   # default: intense sport is log / Last run only
ALERT_ON_GAME_OR_DANCE=0
```

## Face assist (gated)

```bash
ENABLE_FACE_AGGRESSION=0   # default OFF — faces are never analyzed
```

When off, the pipeline records `face_cue_status=disabled` and does not run a face detector.

When on (authorized **lab** use only):

- OpenCV Haar face detect + a conservative geometric contrast proxy, **or**
- `face_aggression=unavailable` if the cascade cannot load (no large FER download is bundled; we do not invent scores)

Face signals may only add an assistive note such as *“possible tense facial expression — verify.”* They must **never**:

- Identify a person
- Infer demographics
- Attach a criminal or “wanted” label

See [ETHICS_AND_SAFETY.md](ETHICS_AND_SAFETY.md).

## Console

**Run Pipeline → Last run** shows:

- Activity class counts (including game / dance that do not page)
- Named sport contexts when present
- Body-aggression max score and cues
- Face cue status (`disabled` / `none_detected` / `unavailable` / `assistive`)
- A per-frame scene-assist table

Alert rationale distinguishes intense sport from confrontation when the policy can tell them apart. Reviewers still decide.

## Limits (again)

This is not production human-activity recognition, not a sports broadcaster, and not a facial affect or identity system. False positives and false negatives are expected. Humans verify every consequential call.
