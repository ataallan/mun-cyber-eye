# Mun Cyber Eye — Architecture (Phase 3 Prototype)

**Company:** Mun Cyber Technologies  
**Product:** Mun Cyber Eye — AI-Powered Human Activity Recognition and Early Threat Detection  
**Tagline:** See danger earlier. Alert faster. Protect people.

## Design principle

> **AI detects and alerts. Humans verify and decide.**

This prototype analyzes **authorized** video sources only. It does **not** perform autonomous enforcement, detention, access control, or legal determinations.

## Pipeline

```
Authorized video / synthetic demo
        ↓
   Frame sampler (ingest/)
        ↓
 Vision adapters (vision/)  ← Phase 3 activity checkpoint if present
                            ← else YOLO if installed
                            ← else honest MOCK
        ↓
   Risk engine (risk/)      ← ordinary | potential_fight | potential_fall | potential_weapon_object
        ↓
 Alert store (alerts/)      ← SQLite + audit log + snapshots
        ↓
 Flask console (app/)       ← session auth · human review actions
        ↓
 Authorized human reviewer
```

## Modules

| Module | Role |
|--------|------|
| `ingest/` | Video file sampling; optional webcam **stub** (disabled by default) |
| `vision/` | Pluggable detectors (`ActivityVisionAdapter`, `YoloVisionAdapter`, `MockVisionAdapter`) plus train/eval |
| `risk/` | Phase 3 category labels or Phase 2 heuristics → `low` / `elevated` / `high` |
| `alerts/` | SQLite persistence, snapshots metadata, audit trail |
| `app/` | Dark-theme Flask console for review |
| `data/activity/` | Labeled train/val/test frames |
| `data/checkpoints/` | Default `activity_demo.joblib` |
| `docs/` | Architecture, ethics & safety, Phase 3 |
| `pipeline.py` | End-to-end orchestration |
| `run.py` | Console entrypoint |

## Alert lifecycle

1. Risk engine sets `should_alert=True` → structured alert created (`status=open`).
2. Operator signs in (minimal session auth).
3. Operator **acknowledges**, **dismisses**, or **escalates** with optional note.
4. Every action is written to `audit_log` with actor + timestamp.

## Vision backends

- **`auto` (default):** Phase 3 activity checkpoint if it loads; else `ultralytics` YOLO; else MOCK.
- **`activity`:** Phase 3 only; MOCK if the checkpoint is missing or invalid.
- **`yolo`:** prefer YOLO; still falls back to MOCK with a warning if unavailable.
- **`mock`:** deterministic scripted detections for demos and CI — fully runnable without a trained model.

MOCK mode remains intentional: the human-in-the-loop demo must work without GPU wheels or a checkpoint.

Phase 3 training and metrics are documented in [PHASE3.md](PHASE3.md).

## Data

- `data/alerts.db` — alerts + audit log  
- `data/snapshots/` — JPEG frames attached to alerts  
- `data/activity/` — labeled activity frames (`train` / `val` / `test`)  
- `data/checkpoints/activity_demo.joblib` — default activity model  
- Paths configurable via `.env`

## Non-goals (Phase 3)

- Production camera fleet management  
- Autonomous lockdown / weapons discharge / facial criminal labeling  
- Large GPU HAR models (optional later; this phase is CPU OpenCV + sklearn)  
- Legal identity or guilt determination  

## Roadmap alignment

Phase 2 = working HITL prototype on controlled sources. Phase 3 = trainable activity categories with documented metrics and fallback. Later phases add secure alert delivery, reviewer-agreement studies, controlled pilots, and lawful integrations — always with human oversight.
