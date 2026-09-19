# Mun Cyber Eye

**Mun Cyber Technologies**  
**AI-Powered Human Activity Recognition and Early Threat Detection**

> See danger earlier. Alert faster. Protect people.

Phase 2 **prototype**: ingest authorized video → sample frames → vision adapters → risk engine → SQLite alerts → Flask console for **human review**.

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
Default demo login (change in `.env`): `operator` / `changeme`

1. Sign in  
2. **Run Pipeline** → Synthetic demo (MOCK)  
3. Review alerts → acknowledge / dismiss / escalate  

### Tests

```bash
pytest -q
```

### Optional YOLO

```bash
pip install ultralytics
# VISION_BACKEND=yolo  or  auto (default)
```

If YOLO is missing, the app uses honest **MOCK** mode and still demos the full alert workflow.

## What this prototype does

| Capability | Status |
|------------|--------|
| Video file ingest + frame sampling | Real (`opencv`) |
| Webcam stub | Stub only (disabled by default) |
| Ultralytics YOLO adapter | Real **if** installed |
| MOCK vision adapter | Real, deterministic demo |
| Risk categories & levels | Heuristic (Phase 2) |
| SQLite alerts + audit log | Real |
| Snapshot attachment | Real |
| Flask dark console + session auth | Real |
| Autonomous enforcement | **Not implemented** (by design) |

### Activity categories

- `ordinary`
- `potential_fight`
- `potential_fall`
- `potential_weapon_object`

### Risk levels

`low` · `elevated` · `high` (with confidence + rationale)

## Layout

```
ingest/          Frame sampler, webcam stub
vision/          YOLO + MOCK adapters
risk/            Heuristic risk engine
alerts/          SQLite store + audit log
app/             Flask console (templates, static)
docs/            ARCHITECTURE.md, ETHICS_AND_SAFETY.md
pipeline.py      End-to-end orchestration
run.py           Entrypoint
tests/           pytest (risk + alerts + mock pipeline)
```

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Ethics and safety](docs/ETHICS_AND_SAFETY.md)

## Mission (from proposal)

MUN Cyber Eye aims to use artificial intelligence and computer vision to provide earlier awareness of potentially dangerous human activities, helping authorized security personnel protect children, communities, and other vulnerable populations before incidents escalate.

## License / use

Prototype for research and controlled demonstration. Deploy only with lawful authorization, privacy safeguards, and human oversight.
