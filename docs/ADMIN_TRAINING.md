# Admin model training — Mun Cyber Eye

Admins can train and activate the Phase 3 activity model from the Flask console. This is the same CPU sklearn path as `python -m vision.train_activity` — a thin wrapper, not a second trainer.

**AI detects and alerts. Humans verify and decide.** Training improves assistive detection only. The system does not enforce, detain, or lock anything.

## Who can train

| Account | Role | Train / activate / upload |
|---------|------|---------------------------|
| Seeded `ADMIN_USERNAME` (default `operator` / `changeme`) | `admin` (via `ADMIN_ROLE`) | Yes |
| `/register` accounts | `operator` | No (status page is readable if they know the URL) |
| Optional `OPERATOR_USERNAME` | `operator` | No |

There is no promote UI in this prototype. To grant training, set `users.role = 'admin'` in `AUTH_DB_PATH` (default `data/auth.db`) for that username, or sign in as the seeded admin.

The **Train models** nav link is shown only when `session.role == admin`.

## Console

1. Sign in as admin.
2. Open **Train models** (`/admin/train`, also `/train`).
3. Review the active checkpoint path, whether it loads, the categories it knows, and last-known val/test metrics stored in the joblib.
4. Review dataset counts under `data/activity/{train,val,test}/<category>/` (the six canonical classes, including `game_or_play` / `dance` / `potential_fight`) and any sport-context or `scene__<place>` folder totals.
5. **Train from labeled data** — model type `forest` or `logreg`, output filename under `data/checkpoints/` (default `activity_custom.joblib`). Overwriting the bundled `activity_demo.joblib` requires the confirmation checkbox. Optional: generate synthetic demo images first if the tree is empty. `--generate-demo` also writes a few `game_or_play__<sport>` folders.
6. After train, the page shows accuracy and per-class precision / recall / F1. Those numbers are written into the checkpoint the same way the CLI does.
7. **Activate checkpoint** writes the chosen file to `data/active_checkpoint.json`. Subsequent **Run Pipeline** / upload runs use that model.
8. Optional **Evaluate** runs `vision.eval_activity` on val or test.
9. **Upload labeled frames** — multi-file into a chosen category (optional sport context writes `game_or_play__<sport>`; optional place type writes `scene__<place>`), or a zip of `train/<category>/*.jpg` including catalog sport folders and `train/scene__street/*.jpg`. Unknown category names and `..` paths are rejected. Sport-context, scene-place, and body-aggression assists stay in the pipeline; training does not replace them. See [SCENE_CONTEXT.md](SCENE_CONTEXT.md).

Training is in-request (seconds on CPU). If a future model exceeds ~60s, switch that job to a background thread and a status file; do not silently hang the worker.

## Active checkpoint pointer

`data/active_checkpoint.json` is **not a secret**. Example:

```json
{
  "path": "/abs/or/relative/data/checkpoints/activity_custom.joblib",
  "activated_at": "2026-09-21T00:00:00Z",
  "activated_by": "operator",
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
| `ACTIVITY_CHECKPOINT` | `data/checkpoints/activity_demo.joblib` | Fallback when no pointer file |

## Honesty

- Synthetic-only training does **not** generalize to real CCTV, lighting, clothing, or camera angles.
- Upload authorized, site-specific examples for the six classes, including game/dance and optional sport or `scene__<place>` folders. Synthetic-only sport courts do **not** mean the model knows most sports or every arena.
- A higher F1 does not authorize skipping human review.
- Train and activate actions are written to `system_audit` (actor = session username).

## CLI (unchanged)

```bash
python -m vision.train_activity --generate-demo --output data/checkpoints/activity_demo.joblib
python -m vision.eval_activity --checkpoint data/checkpoints/activity_custom.joblib --split test
```
