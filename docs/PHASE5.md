# Phase 5 — Live / multi-camera ingest

**Mun Cyber Eye** turns the Phase 4 prototype into a **camera registry + live ingest path**. Detection is automation from those authorized cameras when the pipeline runs. Video file uploads are for **training only**. The risk engine, structured alert schema, notify adapters, and human review console are unchanged.

**AI detects and alerts. Humans verify and decide.**

Authorized cameras only. No autonomous enforcement. Offline feeds and missing credentials are reported on the camera row — the pipeline does **not** invent detections.

## What shipped

| Piece | Location |
|-------|----------|
| Camera registry (`cameras` table) | `ingest/cameras.py` (`CameraStore`) |
| File sampler (unchanged) | `ingest/sampler.py` |
| RTSP open + timeout | `ingest/rtsp.py` |
| Webcam path (`ALLOW_WEBCAM=1`) | `ingest/webcam.py` |
| Source factory | `ingest/source.py` |
| Sequential multi-camera run | `pipeline.run_registered_cameras` |
| Console list / add / edit / disable | `/cameras` (admin / operator) |
| Run Pipeline camera picker | `/run` |
| Dashboard camera health | last_seen / last_error |
| Tests | `tests/test_cameras.py`, `tests/test_ingest_phase5.py`, `tests/test_app_phase5.py` |

Phase 3 activity recognition and Phase 4 structured alerts / Resend / recipients stay in place.

## Camera registry

SQLite (`CAMERA_DB_PATH`, default `data/cameras.db`):

| Column | Meaning |
|--------|---------|
| `id` | Stable id (seeded demos use `demo-file-01`, `demo-rtsp-01`, plus corridor / house / compound / roam stubs) |
| `name` | Operator-facing name |
| `location_label` | Site / zone label stamped onto alerts (free text — not a famous arena name) |
| `place_type` | Optional catalog **or custom** setting (`street`, `roam`, `rooftop_cafe`, …) stamped onto the pipeline. Unknown strings are slugified, not rejected. |
| `source_type` | `file` \| `rtsp` \| `webcam` |
| `uri` | File path, RTSP URL, device index, `MOCK`, or `env:VAR_NAME` |
| `enabled` | Disabled cameras are skipped by “all enabled” |
| `sample_fps` | Target sample rate |
| `sample_interval` | Optional seconds between samples (overrides FPS) |
| `notes` | Authorization scope, SOP, secret name |
| `owner_user_id` / `owner_username` | Display of the first linked account (source of truth is `camera_accounts`) |
| `notify_email` | Optional extra notify addresses for this camera only |
| `camera_accounts` | Many-to-many camera ↔ console account links |
| `last_seen_at` | Last successful ingest |
| `last_error` | Last honest failure (offline, timeout, missing secret, webcam refused) |

On first boot the app seeds the authorized demo set (create-if-missing). Empty `place_type` on those ids is backfilled on list/seed:

1. **Demo Lab File** (`demo-file-01`) — enabled; URI `MOCK` (or `sample_data/demo.mp4` if that clip exists); `place_type=gymnasium`
2. **Authorized RTSP stub** (`demo-rtsp-01`) — **disabled**; URI `env:RTSP_DEMO_URI`; `place_type=street`
3. **Corridor North** (`demo-corridor-01`) — enabled MOCK file; `place_type=corridor_hallway`
4. **House interior demo** (`demo-house-01`) — enabled MOCK file; `place_type=house_interior`
5. **Compound courtyard** (`demo-compound-01`) — enabled MOCK file; `place_type=compound_courtyard`
6. **Roam / patrol cam** (`demo-roam-01`) — enabled MOCK file; `place_type=roam`

## Credentials

- Prefer `env:RTSP_DEMO_URI` (or another env name) so passwords never sit in the URI field.
- If an operator pastes `rtsp://user:pass@host/stream`, templates and public dicts show `rtsp://user:****@host/stream`.
- Edit form: leave URI blank to keep the stored value so a masked password is not written back.

## Ingest behavior

| Source | Behavior |
|--------|----------|
| `file` | Existing OpenCV file sampler. `MOCK` / empty URI yields up to 16 synthetic frames for CI. Missing file → `last_error`, no alerts. Snapshots are named with `camera_id` so sequential multi-camera runs do not overwrite each other. |
| `rtsp` | `cv2.VideoCapture` with `RTSP_CONNECT_TIMEOUT_SEC` (thread join + OpenCV open/read timeouts). Failure → `last_error`, no fake detections. |
| `webcam` | Refused unless `ALLOW_WEBCAM=1`. Default is off with a clear message. |

**Multi-camera:** Run Pipeline still runs one camera by id, or all **enabled** cameras **in sequence**. True parallel streaming is optional later; sequential keeps OpenCV capture simple and failure isolation honest.

**Continuous monitoring** (`monitor.py`) is the customer path. A background worker in the console process sweeps enabled cameras until Stop. Live RTSP and webcam are sampled again every sweep (`MONITOR_MAX_FRAMES_PER_PASS`, then a short pause). Recorded files and `MOCK` URIs are reviewed once per monitoring session unless the file’s size or modification time changes. A dead RTSP URL or missing file sets `last_error` and does not invent detections. `MONITOR_AUTOSTART=1` starts the worker with the process; the console Stop/Start choice is stored in `data/monitor_state.json`.

A suspected incident saves a short clip (`ingest/clips.py`) from the frames already sampled: default 5s before and 5s after the detection (~10s, cap 15s) under `data/clips/`. The alert row stores `clip_path` and the review page plays it. The same camera and category is quiet for `MONITOR_ALERT_COOLDOWN_SEC` (default 60) so a continuous event does not flood alerts. Notify (owner `security_email` / login email and other recipients) still runs when that alert is created. `MAX_INCIDENT_CLIPS` may delete old clip files; it does not delete alert rows.

**Video archive** (`archive_service.py`, `ingest/archive.py`) is optional and off by default (`ARCHIVE_ENABLED=0`). While archive and monitoring are both on, each enabled RTSP, file, or allowed webcam is recorded in short segments under `data/archive/<camera_id>/` (default about 2 minutes). `MOCK` is not archived. **Review** on the Archive page plays segments by camera and by today / 1 day / 5 days. `ARCHIVE_RETENTION_DAYS` (default 1) deletes older segment files only. Disk cost and the split from incident clips are in [PRODUCT_OPS.md](PRODUCT_OPS.md).

Alerts created from a registry camera carry that camera’s `camera_id` and `location_label` (not only `DEFAULT_CAMERA_*` env).

## Console

| Path | Who | Purpose |
|------|-----|---------|
| `/cameras` | admin / operator | List, add, edit, enable / disable (no permanent delete) |
| `/run` | signed-in reviewer | One-shot check of registered cameras, plus optional lab-only MOCK / activity demo. Video files are not a detection path; developers extract frames on Train models. |
| `/` | signed-in reviewer | Alert console, Start / Stop monitoring, archive on/off, camera health |
| `/monitor/start`, `/monitor/stop` | signed-in reviewer | Turn continuous monitoring on or off |
| `/archive` | signed-in reviewer | Play archive segments by camera and time range |
| `/archive/start`, `/archive/stop` | signed-in reviewer | Turn the optional archive on or off |
| `/clips/<file>` | signed-in reviewer | Play an incident clip linked from an alert |

## Safety

- Register only lawful, authorized feeds.
- No facial criminal labeling. Optional face-expression assist is off by default and, when enabled, is unreliable and never identity.
- No enforcement hooks (locks, dispatch, detention) on ingest success or failure.
- A dead RTSP URL is an error on the camera, not a “clear” scene and not a fabricated threat.

## Quick start: register and run

1. Sign in (Create account on a fresh install — no default password).
2. Open **Cameras**. Confirm the seeded Demo Lab File camera (or **Add camera** → type `file`, URI `MOCK` or an authorized path).
3. **Monitoring** starts with the console when `MONITOR_AUTOSTART=1`. Use **Start monitoring** / **Stop monitoring** on the alert console. **Run** is a one-shot check, not the customer path. Lab-only MOCK is not the customer path. To label a clip, a Mun Cyber developer uses **Train models → Extract frames from video**. Customers use shipped checkpoints.
4. Open an alert: `camera_id` and `location_label` match the registry row. If monitoring raised it, play the short incident clip on the review page. Acknowledge / dismiss / escalate / reopen — the row is retained.
5. To try RTSP: set `RTSP_DEMO_URI` in `.env`, enable the stub camera. A bad or empty URI marks `last_error` and creates no alerts.

```bash
# .env
ALLOW_WEBCAM=0
RTSP_CONNECT_TIMEOUT_SEC=8
# RTSP_DEMO_URI=rtsp://…   # authorized feed only
```
