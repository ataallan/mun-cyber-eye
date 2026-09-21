# Developer model training — Mun Cyber Eye

Mun Cyber **developer** accounts can train and activate the Phase 3 activity model from the Flask console. This is the same CPU sklearn path as `python -m vision.train_activity` — a thin wrapper, not a second trainer.

Customer **site admin** and **operator** accounts run detection and review on **shipped checkpoints**. They do not see **Train models** and cannot call train / activate / upload / extract APIs.

**AI detects and alerts. Humans verify and decide.** Training improves assistive detection only. The system does not enforce, detain, or lock anything.

## Who can train

| Account | Role | Train / activate / upload / extract |
|---------|------|--------------------------------------|
| `DEVELOPER_USERNAME` only when **both** username and password are set in `.env` (lab machines; no published default) | `developer` | Yes |
| First `/register` account on a fresh install | `admin` (site ops) | No |
| Optional `ADMIN_USERNAME` when username **and** password are set | `admin` (or `operator` if `ADMIN_ROLE=operator`) | No |
| Later `/register` accounts | `operator` | No |
| Optional `OPERATOR_USERNAME` when username **and** password are set | `operator` | No |

Create account **never** grants `developer`. Customer `.env.example` leaves `DEVELOPER_*` empty. Mun Cyber staff set those variables on lab machines only — never `operator` / `changeme`.

The **Train models** nav link and `/admin/train` (GET and mutating POST) are shown / allowed only when `session.role == developer`.

## Console

1. Sign in as the env-seeded developer on a lab machine.
2. Open **Train models** (`/admin/train`, also `/train`).
3. Review the active checkpoint path, whether it loads, the categories it knows, and last-known val/test metrics stored in the joblib.
4. Review dataset counts under `data/activity/{train,val,test}/<category>/` (the six canonical classes, including `game_or_play` / `dance` / `potential_fight`) and any sport-context or `scene__<place>` folder totals. Object-class counts (sibling `data/objects/{train,val,test}/<object_id>/`) are listed separately and do not feed the activity trainer.
5. **Train from labeled data** — model type `forest` or `logreg`, output filename under `data/checkpoints/` (default `activity_custom.joblib`). Overwriting the bundled `activity_demo.joblib` requires the confirmation checkbox. Optional: generate synthetic demo images first if the tree is empty. `--generate-demo` also writes a few `game_or_play__<sport>` folders.
6. After train, the page shows accuracy and per-class precision / recall / F1. Those numbers are written into the checkpoint the same way the CLI does.
7. **Activate checkpoint** writes the chosen file to `data/active_checkpoint.json`. Subsequent **Run Pipeline** camera runs use that model.
8. Optional **Evaluate** runs `vision.eval_activity` on val or test.
9. **Upload labeled frames** — multi-file into a chosen category (optional sport context writes `game_or_play__<sport>`; optional place type writes `scene__<place>`), or a zip of `train/<category>/*.jpg` including catalog sport folders and `train/scene__street/*.jpg`. Unknown category names and `..` paths are rejected. Sport-context, scene-place, and body-aggression assists stay in the pipeline; training does not replace them. See [SCENE_CONTEXT.md](SCENE_CONTEXT.md). Uploads remain **training-only and developer-only**.
10. **Extract frames from video** — **training only**, not live detection. Developer picks kind (activity / game_or_play+sport / scene place / **object class** / **dangerous / weapon-like class**), uploads an authorized mp4 / avi / mov / mkv, and sets sample FPS / max frames. `FrameSampler` writes JPEGs into `data/activity/train/...`, `data/objects/train/<object_id>/`, or `data/dangerous/train/<id>/`. Activity class `potential_weapon_object` remains a six-class trainer folder. **Videos are sampled to frames; the sklearn activity trainer still learns from images.** After extract, use **Train from labeled data** for activity classes. Optional **Train object model** fits a small CPU classifier when enough object images exist; runtime inventory still prefers YOLO and will not invent a refrigerator when the detector is missing. See [OBJECTS_AND_STRUCTURES.md](OBJECTS_AND_STRUCTURES.md) and [DANGEROUS_OBJECTS.md](DANGEROUS_OBJECTS.md).

Training is in-request (seconds on CPU). If a future model exceeds ~60s, switch that job to a background thread and a status file; do not silently hang the worker.

## Active checkpoint pointer

`data/active_checkpoint.json` is **not a secret**. Example:

```json
{
  "path": "/abs/or/relative/data/checkpoints/activity_custom.joblib",
  "activated_at": "2026-09-21T00:00:00Z",
  "activated_by": "labdev",
  "source": "console"
}
```

Resolution order used by `create_app` and `vision.detector.create_adapter`:

1. `data/active_checkpoint.json` (`ACTIVE_CHECKPOINT_FILE`) if present
2. `ACTIVITY_CHECKPOINT` env / `.env`
3. Bundled `data/checkpoints/activity_demo.joblib`

Do not put API keys in this file. Changing `.env` is unnecessary for a console swap.

Paths:

| Config | Default | Purpose |
|--------|---------|---------|
| `ACTIVE_CHECKPOINT_FILE` | `data/active_checkpoint.json` | Durable pointer |
| `CHECKPOINTS_DIR` | `data/checkpoints` | Train output + activate list |
| `ACTIVITY_DATA_ROOT` | `data/activity` | Labeled frames |
| `OBJECTS_DATA_ROOT` | `data/objects` | Labeled object/structure frames (sibling tree) |
| `DANGEROUS_DATA_ROOT` | `data/dangerous` | Labeled dangerous / weapon-like frames (sibling tree) |
| `ACTIVITY_CHECKPOINT` | `data/checkpoints/activity_demo.joblib` | Fallback when no pointer file |

## Honesty

- Synthetic-only training does **not** generalize to real CCTV, lighting, clothing, or camera angles.
- Upload authorized, site-specific examples for the six classes, including game/dance and optional sport or `scene__<place>` folders. Synthetic-only sport courts do **not** mean the model knows most sports or every arena. Object-class videos become JPEG frames under `data/objects/`; they do not train the activity model by themselves.
- A higher F1 does not authorize skipping human review.
- Train and activate actions are written to `system_audit` (actor = session username).
- **Videos are sampled to frames; the sklearn activity trainer still learns from images.**
- Fall / gunshot / aimed-firearm / thrown-object / dangerous-object intensity assists are pipeline extras, not extra sklearn classes. `potential_gunshot` is a risk enum only. See [GUNSHOTS_AND_FALLS.md](GUNSHOTS_AND_FALLS.md) and [DANGEROUS_OBJECTS.md](DANGEROUS_OBJECTS.md).

## CLI (unchanged)

Lab / developer machines can still train from the CLI:

```bash
python -m vision.train_activity --generate-demo --output data/checkpoints/activity_demo.joblib
python -m vision.eval_activity --checkpoint data/checkpoints/activity_custom.joblib --split test
```
