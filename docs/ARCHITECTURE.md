# Mun Cyber Eye — Architecture (Phase 2 Prototype)

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
 Vision adapters (vision/)  ← YOLO if installed, else honest MOCK
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
| `vision/` | Pluggable detectors (`YoloVisionAdapter`, `MockVisionAdapter`) |
| `risk/` | Heuristic risk levels: `low` / `elevated` / `high` + confidence |
| `alerts/` | SQLite persistence, snapshots metadata, audit trail |
| `app/` | Dark-theme Flask console for review |
| `docs/` | Architecture, ethics & safety |
| `pipeline.py` | End-to-end orchestration |
| `run.py` | Console entrypoint |

## Alert lifecycle

1. Risk engine sets `should_alert=True` → structured alert created (`status=open`).
2. Operator signs in (minimal session auth).
3. Operator **acknowledges**, **dismisses**, or **escalates** with optional note.
4. Every action is written to `audit_log` with actor + timestamp.

## Vision backends

- **`auto` (default):** try `ultralytics` YOLO; on failure, use MOCK.
- **`yolo`:** prefer YOLO; still falls back to MOCK with a warning if unavailable.
- **`mock`:** deterministic scripted detections for demos and CI — fully runnable without ML wheels.

MOCK mode is intentional: Phase 2 prioritizes a working human-in-the-loop demo over brittle dependency chains.

## Data

- `data/alerts.db` — alerts + audit log  
- `data/snapshots/` — JPEG frames attached to alerts  
- Paths configurable via `.env`

## Non-goals (Phase 2)

- Production camera fleet management  
- Autonomous lockdown / weapons discharge / facial criminal labeling  
- Training large HAR models (Phase 3+)  
- Legal identity or guilt determination  

## Roadmap alignment

Phase 2 = prototype on controlled / operator-supplied sources. Later phases add trained activity models, secure alert delivery, metrics, controlled pilots, and lawful integrations — always with human oversight.
