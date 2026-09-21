# Phase 3b — Game, dance, and confrontation

**AI detects and alerts. Humans verify and decide.**

Phase 3 originally had four activity classes. Energetic motion (sports, roughhousing, dance) can look like a fight to both heuristics and a tiny demo model. This increment adds two **non-threat** classes so the operator can see the difference without paging a confrontation alert.

## What changed

| Canonical label | Plain language | Default alert |
|-----------------|----------------|---------------|
| `ordinary` | ordinary | No |
| `game_or_play` | game or play | No |
| `dance` | dance | No |
| `potential_fight` | potential confrontation | Yes |
| `potential_fall` | potential fall | Yes |
| `potential_weapon_object` | potential weapon-like object | Yes |

Confrontation aliases accepted by the loader and risk engine:

- `confrontation` → `potential_fight`
- `fight` → `potential_fight`
- `altercation` → `potential_fight`

`ordinary`, `game_or_play`, and `dance` are **predict / log only** by default (`should_alert=False`). Set `ALERT_ON_GAME_OR_DANCE=1` only if operators explicitly want those classes to create alerts too.

The Run Pipeline **Last run** panel lists every predicted category (including game and dance) even when no alert is created. Alert detail and the dashboard show the same names in plain language.

Rationale text states clearly when the model thinks the scene is game/play or dance **rather than a fight**. Safety banner is unchanged: AI detects and alerts; humans verify and decide. No enforcement.

## Honest limits

- The bundled `activity_demo.joblib` is still trained on **synthetic geometric scenes** (green field + ball for play, magenta arcs for dance, red streaks for fight). It will **not** generalize to real CCTV until you retrain on labeled site video.
- Game vs fight is hard even for humans: clothing, camera angle, and intent are not in the feature vector. Treat every confrontation alert as provisional.
- Phase 2 MOCK heuristics can still flag `rapid_motion` + multiple people as a fight. The activity path and the scripted MOCK game/dance frames emit the new labels so the console can demonstrate differentiation.

Retrain:

```bash
python -m vision.train_activity --generate-demo --overwrite-demo \
  --output data/checkpoints/activity_demo.joblib
```
