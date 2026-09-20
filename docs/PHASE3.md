# Phase 3 — Activity Recognition

**Mun Cyber Eye** trains and evaluates a lightweight activity classifier for the proposal categories, then loads the checkpoint in the existing human-in-the-loop pipeline.

**AI detects and alerts. Humans verify and decide.**

## What shipped

| Piece | Location |
|-------|----------|
| Dataset layout | `data/activity/{train,val,test}/<category>/` |
| Feature extraction | `vision/features.py` (OpenCV, CPU) |
| Train | `python -m vision.train_activity` |
| Evaluate | `python -m vision.eval_activity` |
| Adapter + checkpoint I/O | `vision/activity.py` |
| Default demo checkpoint | `data/checkpoints/activity_demo.joblib` |
| Pipeline fallback | `vision.detector.create_adapter` |

The Flask console, SQLite alerts, and reviewer actions (acknowledge / dismiss / escalate) are unchanged. Phase 3 only adds a trainable vision path in front of the same risk + alert contract.

## Categories

Canonical labels (proposal / Phase 2):

- `ordinary`
- `potential_fight`
- `potential_fall`
- `potential_weapon_object`

Related informal folder names are accepted when loading data (`fight`, `fall`, `weapon`, `person_down`, …) and mapped onto those four classes. Phase 2 heuristic signals (`close_proximity`, `raised_object`, …) remain available when the activity model is **not** loaded.

## Approach (CPU, no GPU)

OpenCV handcrafted features + scikit-learn:

1. Resize frame to 64×48.
2. HSV color histogram, HOG-like gradient histograms, edge/geometry stats.
3. Optional motion stats vs the previous frame (frame stack of 2).
4. `StandardScaler` + `RandomForestClassifier` (or `--model logreg`).

This is small enough to train on a laptop in seconds. A tiny PyTorch CNN on frame stacks is a reasonable later swap; it is **not** required to run Phase 3.

## Honest limits

- The bundled `activity_demo.joblib` is trained on **synthetic geometric scenes**, not authorized CCTV.
- Color/HOG/geometry cues will **not** generalize to real hallways, lighting, clothing, or camera angles until you retrain on operator-supplied labeled video.
- There is no pose estimator, tracker, or temporal transformer. Motion is a cheap frame-difference.
- `potential_weapon_object` is a **visual proxy** (bright object near a figure), not weapon identification and not proof of possession or intent.
- False positives and false negatives are expected. Every non-ordinary output is an **alert for a human**, never an enforcement action.

## Dataset layout

```
data/activity/
  train/
    ordinary/*.jpg
    potential_fight/*.jpg
    potential_fall/*.jpg
    potential_weapon_object/*.jpg
  val/    (same class folders)
  test/   (same class folders)
```

Supported files: `.jpg`, `.jpeg`, `.png`, `.bmp`. Put your own authorized frames in those folders (do not commit private video).

Create empty folders and/or a synthetic demo set:

```bash
python -m vision.train_activity --generate-demo \
  --data-root data/activity \
  --output data/checkpoints/activity_demo.joblib
```

`--generate-demo` writes simple painted scenes so CI and first-time clones can train without a camera. Use `--overwrite-demo` to regenerate.

## Train

```bash
python -m vision.train_activity \
  --data-root data/activity \
  --output data/checkpoints/activity_demo.joblib \
  --model forest \
  --seed 7
```

Options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--data-root` | `data/activity` | Split/category tree |
| `--output` | `data/checkpoints/activity_demo.joblib` | Checkpoint path |
| `--model` | `forest` | `forest` or `logreg` |
| `--generate-demo` | off | Synthesize demo images |
| `--n-train` / `--n-val` / `--n-test` | 40 / 12 / 12 | Images per class when generating |
| `--seed` | 7 | Reproducibility |

Training prints val/test **accuracy** and **per-class precision, recall, F1** when those splits exist, and stores the same numbers inside the checkpoint.

## Evaluate

```bash
python -m vision.eval_activity \
  --data-root data/activity \
  --checkpoint data/checkpoints/activity_demo.joblib \
  --split test
```

Optional JSON report:

```bash
python -m vision.eval_activity --split test --json data/checkpoints/activity_demo_metrics.json
```

Report contents:

- overall accuracy
- per-class precision / recall / F1 / support
- macro averages
- confusion matrix (rows = true, cols = predicted)

## Default checkpoint and pipeline

The bundled demo checkpoint, trained on `--generate-demo` images, scores **1.00 accuracy / macro F1 on its synthetic holdout**. That measures separability of the painted scenes, not real camera performance.

Default path (overridable):

```
ACTIVITY_CHECKPOINT=data/checkpoints/activity_demo.joblib
```

`VISION_BACKEND` selection:

| Value | Behavior |
|-------|----------|
| `auto` (default) | Phase 3 checkpoint if it loads → else YOLO → else MOCK |
| `activity` | Phase 3 only; MOCK if the checkpoint is missing or invalid |
| `yolo` | Ultralytics YOLO + Phase 2 heuristics; MOCK if YOLO is missing |
| `mock` | Deterministic scripted detections (Phase 2 demo) |

The risk engine treats a detection whose label is one of the four categories as a Phase 3 decision. Otherwise it uses the Phase 2 label heuristics. Alerts still require a human to acknowledge, dismiss, or escalate.

Console:

1. **Run Pipeline → Phase 3 activity model** — synthetic class scenes through the checkpoint (or MOCK fallback).
2. **Synthetic demo (MOCK)** — unchanged Phase 2 HITL walkthrough.
3. **Authorized video upload** — `create_adapter()` (activity when the checkpoint is present).

## Retrain on real authorized data

1. Sample frames from operator-approved video (`ingest/sampler.py` or any JPEG export).
2. Label into the four class folders under `data/activity/train` (and val/test).
3. Retrain with the same command (omit `--generate-demo`).
4. Point `ACTIVITY_CHECKPOINT` at the new `.joblib`.
5. Review metrics **and** reviewer agreement before any pilot. Do not treat a higher F1 as authorization to skip humans.
