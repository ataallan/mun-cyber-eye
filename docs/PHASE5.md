# Phase 5 — Live / multi-camera ingest

**Mun Cyber Eye** turns the Phase 4 “run once on a video or MOCK” prototype into a **camera registry + live ingest path**. The risk engine, structured alert schema, notify adapters, and human review console are unchanged.

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
| `id` | Stable id (seeded demos use `demo-file-01`, `demo-rtsp-01`) |
| `name` | Operator-facing name |
| `location_label` | Site / zone label stamped onto alerts (free text — not a famous arena name) |
| `place_type` | Optional catalog setting (`street`, `corridor_hallway`, `house_interior`, …) stamped onto the pipeline |
| `source_type` | `file` \| `rtsp` \| `webcam` |
| `uri` | File path, RTSP URL, device index, `MOCK`, or `env:VAR_NAME` |
| `enabled` | Disabled cameras are skipped by “all enabled” |
| `sample_fps` | Target sample rate |
| `sample_interval` | Optional seconds between samples (overrides FPS) |
| `notes` | Authorization scope, SOP, secret name |
| `last_seen_at` | Last successful ingest |
| `last_error` | Last honest failure (offline, timeout, missing secret, webcam refused) |

On first boot, if the table is empty, the app seeds:

1. **Demo Lab File** (`demo-file-01`) — enabled; URI `MOCK` (or `sample_data/demo.mp4` if that clip exists)
2. **Authorized RTSP stub** (`demo-rtsp-01`) — **disabled**; URI `env:RTSP_DEMO_URI`

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

**Multi-camera:** the prototype runs one camera by id, or all **enabled** cameras **in sequence**. True parallel streaming is optional later; sequential keeps OpenCV capture simple and failure isolation honest.

Alerts created from a registry camera carry that camera’s `camera_id` and `location_label` (not only `DEFAULT_CAMERA_*` env).

## Console

| Path | Who | Purpose |
|------|-----|---------|
| `/cameras` | admin / operator | List, add, edit, enable / disable |
| `/run` | signed-in reviewer | MOCK, activity demo, one camera, all enabled, or upload. Attaching a video file auto-selects **Authorized video file upload** (server-side if a filename is present, so Synthetic MOCK cannot run silently). Zero-alert runs still report frames processed. Last run lists optional home/community objects when YOLO is available; otherwise `objects_backend=unavailable` (nothing invented). |
| `/` | signed-in reviewer | Alert console + camera health |

## Safety

- Register only lawful, authorized feeds.
- No facial criminal labeling. Optional face-expression assist is off by default and, when enabled, is unreliable and never identity.
- No enforcement hooks (locks, dispatch, detention) on ingest success or failure.
- A dead RTSP URL is an error on the camera, not a “clear” scene and not a fabricated threat.

## Quick start: register and run

1. Sign in as `admin` or `operator`.
2. Open **Cameras**. Confirm the seeded Demo Lab File camera (or **Add camera** → type `file`, URI `MOCK` or an authorized path).
3. **Run Pipeline** → Registered camera → that camera (or “All enabled”). For an authorized clip, select **Authorized video file upload** or attach the file (upload is auto-selected so Synthetic MOCK cannot run silently).
4. Open an alert: `camera_id` and `location_label` match the registry row.
5. To try RTSP: set `RTSP_DEMO_URI` in `.env`, enable the stub camera. A bad or empty URI marks `last_error` and creates no alerts.

```bash
# .env
ALLOW_WEBCAM=0
RTSP_CONNECT_TIMEOUT_SEC=8
# RTSP_DEMO_URI=rtsp://…   # authorized feed only
```
