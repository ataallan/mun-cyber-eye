# Mun Cyber Eye — Architecture (Phase 5 Prototype)

**Company:** Mun Cyber Technologies  
**Product:** Mun Cyber Eye — AI-Powered Human Activity Recognition and Early Threat Detection  
**Tagline:** See danger earlier. Alert faster. Protect people.

## Design principle

> **AI detects and alerts. Humans verify and decide.**

This prototype analyzes **authorized** video sources only. It does **not** perform autonomous enforcement, detention, access control, or legal determinations. Outbound email and webhooks notify authorized personnel; they do not close a review or act on anyone.

## Pipeline

```
Camera registry (SQLite)  →  authorized file / RTSP / webcam / MOCK
        ↓
   Frame sampler (ingest/)   sequential per camera in this prototype
        ↓
 Vision adapters (vision/)  ← Phase 3 activity checkpoint if present
                            ← else YOLO if installed
                            ← else honest MOCK
        ↓
   Risk engine (risk/)      ← ordinary | game_or_play | dance | potential_fight | potential_fall | potential_weapon_object
        ↓
 Alert store (alerts/)      ← SQLite + structured schema + camera_id + location_label
        ↓
 Notify adapters (alerts/)  ← console (primary) · Resend email · optional webhook
        ↓
 Flask console (app/)       ← SQLite users · cameras · human review · recipients
        ↓
 Authorized human reviewer
```

## Modules

| Module | Role |
|--------|------|
| `ingest/` | Camera registry; file sampler; RTSP with timeout; webcam gated by `ALLOW_WEBCAM` |
| `vision/` | Pluggable detectors (`ActivityVisionAdapter`, `YoloVisionAdapter`, `MockVisionAdapter`) plus train/eval |
| `risk/` | Phase 3 category labels or Phase 2 heuristics → `low` / `elevated` / `high` |
| `alerts/` | SQLite persistence, structured payload, recipients, delivery adapters |
| `app/` | Dark-theme Flask console for review and authorized notification |
| `data/activity/` | Labeled train/val/test frames |
| `data/checkpoints/` | Default `activity_demo.joblib` |
| `docs/` | Architecture, ethics & safety, Phase 3, Phase 4, Phase 5 |
| `pipeline.py` | End-to-end orchestration |
| `run.py` | Console entrypoint |

## Alert lifecycle

1. Risk engine sets `should_alert=True` → structured alert created (`status=open`, `delivery_status=pending`).
2. Optional outbound delivery: Resend email to authorized recipients and/or SIEM webhook. Missing keys are `queued` / `undelivered` — never reported as sent.
3. Operator signs in (SQLite users; env admin is seeded on boot). `admin` or `operator` may manage recipients and resend. See [AUTH.md](AUTH.md).
4. Operator **acknowledges**, **dismisses**, or **escalates** with optional note.
5. Every review action and notify/resend attempt is written to `audit_log` with actor + timestamp. Channel attempts go to `delivery_log`.

Structured fields and delivery statuses are documented in [PHASE4.md](PHASE4.md).

## Vision backends

- **`auto` (default):** Phase 3 activity checkpoint if it loads; else `ultralytics` YOLO; else MOCK.
- **`activity`:** Phase 3 only; MOCK if the checkpoint is missing or invalid.
- **`yolo`:** prefer YOLO; still falls back to MOCK with a warning if unavailable.
- **`mock`:** deterministic scripted detections for demos and CI — fully runnable without a trained model.

MOCK mode remains intentional: the human-in-the-loop demo must work without GPU wheels, a checkpoint, or a Resend key.

Phase 3 training and metrics are documented in [PHASE3.md](PHASE3.md).

## Data

- `data/alerts.db` — alerts + audit log + delivery log + operators  
- `data/auth.db` — console login accounts (admin / operator)  
- `data/cameras.db` — authorized camera registry + last_seen / last_error  
- `data/snapshots/` — JPEG frames attached to alerts  
- `data/uploads/` — operator-supplied authorized video files (gitignored)  
- `data/activity/` — labeled activity frames (`train` / `val` / `test`)  
- `data/checkpoints/activity_demo.joblib` — default activity model  
- Paths configurable via `.env`

## Non-goals (Phase 5)

- True parallel live streaming of a camera fleet (sequential ingest is the prototype)  
- Autonomous lockdown / weapons discharge / facial criminal labeling  
- Large GPU HAR models (optional later; this phase is CPU OpenCV + sklearn)  
- Legal identity or guilt determination  
- Guaranteed third-party email delivery (we report the provider result honestly)  
- Connecting to unauthorized cameras

## Roadmap alignment

Phase 2 = working HITL prototype on controlled sources. Phase 3 = trainable activity categories with documented metrics and fallback. Phase 4 = structured alerts and secure notification to authorized personnel. Phase 5 = camera registry and live ingest (file / RTSP / gated webcam) that still feeds human review. Later phases add reviewer-agreement studies, controlled pilots, and lawful integrations — always with human oversight.

Live ingest details: [PHASE5.md](PHASE5.md).
