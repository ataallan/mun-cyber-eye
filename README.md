# Mun Cyber Eye

**Mun Cyber Technologies**  
**AI-Powered Human Activity Recognition and Early Threat Detection**

> See danger earlier. Alert faster. Protect people.

Phase 4 **prototype**: ingest authorized video → sample frames → **activity model** (or YOLO / MOCK) → risk engine → structured SQLite alerts → Flask console for **human review** → optional Resend email / SIEM webhook to **authorized personnel**.

**Safety banner (always on):** *AI detects and alerts. Humans verify and decide.*

Authorized cameras only. No autonomous enforcement.

## Quick start

```bash
cd mun-cyber-eye
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py
```

Open **http://127.0.0.1:5055**

Sign in with either:

- **Create account** on the login page (new operators), or  
- The seeded demo admin (change in `.env`): `operator` / `changeme`

1. Sign in  
2. **Run Pipeline** → Synthetic demo (MOCK) or **Phase 3 activity model**  
3. Open an alert → structured payload, delivery status, acknowledge / dismiss / escalate  
4. Optional: **Recipients** + `RESEND_API_KEY` to email authorized operators  

Without a Resend key the console still works. Delivery is marked `queued` / `undelivered` — never reported as sent.

Forgot password: if `RESEND_API_KEY` and `RESEND_FROM` are set, a time-limited reset link is emailed. If they are not set, the console says email is not configured (it will not claim a message was sent). Local demos can read the reset URL from the application log, or set `AUTH_SHOW_RESET_URL=1`. Details: [docs/AUTH.md](docs/AUTH.md).

### Tests

```bash
pytest -q
```

### Phase 4 alert delivery

Console review is primary. Email and webhook are pluggable delivery channels.

```bash
# .env — never invent success if the key is empty
RESEND_API_KEY=re_xxxxxxxx
RESEND_FROM=Mun Cyber Technologies <info@muncyber.com>
ALERT_EMAIL_RECIPIENTS=you@your-domain.com
# optional SIEM/SOAR
ALERT_WEBHOOK_URL=
```

The Resend adapter uses stdlib `urllib` with an explicit `User-Agent` so Cloudflare does not block the request. Full demo steps: [docs/PHASE4.md](docs/PHASE4.md).

### Phase 3 activity recognition

Lightweight **OpenCV features + scikit-learn** (no GPU). A demo checkpoint ships at `data/checkpoints/activity_demo.joblib`. If that file is missing or fails to load, the pipeline falls back to Phase 2 YOLO heuristics or honest MOCK.

```bash
# Train (synthetic demo set if you have no labeled frames)
python -m vision.train_activity --generate-demo \
  --output data/checkpoints/activity_demo.joblib

# Evaluate: accuracy + per-class precision / recall / F1
python -m vision.eval_activity \
  --checkpoint data/checkpoints/activity_demo.joblib \
  --split test
```

`VISION_BACKEND=auto` (default) uses the Phase 3 checkpoint when present, then YOLO, then MOCK.  
`VISION_BACKEND=mock` keeps the Phase 2 scripted demo.  
`ACTIVITY_CHECKPOINT` overrides the default path. Full notes: [docs/PHASE3.md](docs/PHASE3.md).

### Optional YOLO

```bash
pip install ultralytics
# VISION_BACKEND=yolo
```

If YOLO is missing and no activity checkpoint loads, the app uses honest **MOCK** mode and still demos the full alert workflow.

## What this prototype does

| Capability | Status |
|------------|--------|
| Video file ingest + frame sampling | Real (`opencv`) |
| Webcam stub | Stub only (disabled by default) |
| Phase 3 activity model (OpenCV + sklearn) | Real **if** checkpoint loads |
| Train / eval (accuracy, P/R/F1 per class) | Real (`python -m vision.train_activity` / `eval_activity`) |
| Demo activity checkpoint | Bundled (`data/checkpoints/activity_demo.joblib`, synthetic data) |
| Ultralytics YOLO adapter | Real **if** installed |
| MOCK vision adapter | Real, deterministic demo |
| Risk categories & levels | Phase 3 model labels, else Phase 2 heuristics |
| Structured SQLite alerts + audit log | Real |
| Snapshot attachment | Real |
| Flask dark console + session auth | Real (SQLite users, register, forgot/reset password) |
| Authorized recipient directory | Real (`.env` + `operators` table) |
| Resend email / webhook delivery | Real **if** configured; otherwise honest `queued` / `undelivered` |
| Delivery + ack tracking | Real (`delivery_log` + Phase 2 audit) |
| Autonomous enforcement | **Not implemented** (by design) |

### Activity categories

- `ordinary`
- `potential_fight`
- `potential_fall`
- `potential_weapon_object`

### Risk levels

`low` · `elevated` · `high` (with confidence + rationale)

Responder-facing **severity**: `info` · `warning` · `critical`

## Layout

```
ingest/          Frame sampler, webcam stub
vision/          Activity + YOLO + MOCK adapters; train/eval
risk/            Risk engine (model labels or heuristics)
alerts/          SQLite store, structured schema, notify adapters
app/             Flask console (templates, static, auth)
data/activity/   Labeled frames (train/val/test/<category>)
data/checkpoints/activity_demo.joblib
docs/            ARCHITECTURE.md, ETHICS_AND_SAFETY.md, PHASE3.md, PHASE4.md
pipeline.py      End-to-end orchestration
run.py           Entrypoint
tests/           pytest (risk + alerts + notify + mock + activity)
```

## Docs

- [Phase 4 — alert system](docs/PHASE4.md)
- [Phase 3 — activity recognition](docs/PHASE3.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Authentication](docs/AUTH.md)
- [Ethics and safety](docs/ETHICS_AND_SAFETY.md)

## Mission (from proposal)

MUN Cyber Eye aims to use artificial intelligence and computer vision to provide earlier awareness of potentially dangerous human activities, helping authorized security personnel protect children, communities, and other vulnerable populations before incidents escalate.

## License / use

Prototype for research and controlled demonstration. Deploy only with lawful authorization, privacy safeguards, and human oversight.
