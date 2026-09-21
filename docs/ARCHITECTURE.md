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
   Risk engine (risk/)      ← ordinary | game_or_play (+ optional sport_context) | dance | potential_fight | potential_fall | potential_weapon_object | potential_gunshot (video proxy, not a sklearn class)
                            ← scene place_type + kit-color assists; optional YOLO object inventory; body-aggression proxies; optional gated face assist (off by default)
                            ← fall manner; gunshot video proxy (audio off by default); aimed-firearm geometry; thrown-object-toward-person
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
| `vision/` | Pluggable detectors (`ActivityVisionAdapter`, `YoloVisionAdapter`, `MockVisionAdapter`) plus train/eval, sports catalog, scene-place / kit assists, object/structure catalog + YOLO inventory, aggression / optional face assists, fall manner, gunshot video proxy, aimed-firearm geometry, thrown-object assist |
| `risk/` | Phase 3 category labels or Phase 2 heuristics → `low` / `elevated` / `high`; sport + setting vs fight policy |
| `alerts/` | SQLite persistence, structured payload, recipients, delivery adapters |
| `app/` | Dark-theme Flask console for review and authorized notification |
| `data/activity/` | Labeled train/val/test frames |
| `data/checkpoints/` | Default `activity_demo.joblib` |
| `docs/` | Architecture, ethics & safety, product ops, standalone Windows, Phase 3, sports/aggression, scene context, objects, gunshots/falls/aimed/thrown, Phase 4, Phase 5 |
| `pipeline.py` | End-to-end orchestration |
| `run.py` | Console entrypoint |

## Alert lifecycle

1. Risk engine sets `should_alert=True` → structured alert created (`status=open`, `delivery_status=pending`).
2. Optional outbound delivery: Resend email to authorized recipients and/or SIEM webhook. Missing keys are `queued` / `undelivered` — never reported as sent.
3. Operator signs in (SQLite users; first Create account is site admin; optional env bootstrap only if both username and password are set). `admin` or `operator` may manage recipients and resend. See [AUTH.md](AUTH.md).
4. Operator **acknowledges**, **dismisses**, **escalates**, or **reopens** with optional note. There is no delete that wipes the alert. Mun Cyber **developer** accounts may train or activate a checkpoint (`/admin/train`). Customer site admins cannot.
5. Every review action and notify/resend attempt is written to `audit_log` with actor + timestamp. Channel attempts go to `delivery_log`. Train / activate / labeled-frame uploads go to `system_audit`.

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
- `data/uploads/` — leftover train-extract source files may sit here (gitignored); not a customer video library  
- `data/activity/` — labeled activity frames (`train` / `val` / `test`)  
- `data/objects/` — labeled object/structure frames (sibling of activity)  
- `data/checkpoints/activity_demo.joblib` — default activity model  
- `data/active_checkpoint.json` — console-activated checkpoint pointer (not a secret)  
- Paths configurable via `.env`

## Non-goals (Phase 5)

- True parallel live streaming of a camera fleet (sequential ingest is the prototype)  
- Autonomous lockdown / weapons discharge / facial criminal labeling (face assist, when enabled, is unreliable and never identity)  
- Claiming the activity model “knows most sports,” understands all sports arenas, or can read facial aggression reliably
- Large GPU HAR models (optional later; this phase is CPU OpenCV + sklearn)  
- Legal identity or guilt determination  
- Guaranteed third-party email delivery (we report the provider result honestly)  
- Connecting to unauthorized cameras

## Roadmap alignment

Phase 2 = working HITL prototype on controlled sources. Phase 3 = trainable activity categories with documented metrics and fallback. Phase 4 = structured alerts and secure notification to authorized personnel. Phase 5 = camera registry and live ingest (file / RTSP / gated webcam) that still feeds human review. Later phases add reviewer-agreement studies, controlled pilots, and lawful integrations — always with human oversight.

Live ingest details: [PHASE5.md](PHASE5.md). Sports / aggression assists: [SPORTS_AND_AGGRESSION.md](SPORTS_AND_AGGRESSION.md). Scene / setting assist: [SCENE_CONTEXT.md](SCENE_CONTEXT.md). Object / structure inventory: [OBJECTS_AND_STRUCTURES.md](OBJECTS_AND_STRUCTURES.md). Dangerous objects / use intensity: [DANGEROUS_OBJECTS.md](DANGEROUS_OBJECTS.md). Gunshots, falls, aimed firearms, thrown objects: [GUNSHOTS_AND_FALLS.md](GUNSHOTS_AND_FALLS.md).
