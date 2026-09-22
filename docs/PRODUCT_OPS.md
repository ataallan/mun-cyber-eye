# Product operations and retention — Mun Cyber Eye

**AI detects and alerts. Humans verify and decide.**

This note records owner product rules for Mun Cyber Technologies. See also [ETHICS_AND_SAFETY.md](ETHICS_AND_SAFETY.md) and [STANDALONE.md](STANDALONE.md).

## Detection path

Elevated-risk / violent-conduct detection is **automation from authorized cameras** in the registry:

- File-source cameras
- RTSP cameras
- Webcam only if `ALLOW_WEBCAM=1`

**Continuous monitoring** is the customer path. While monitoring is on, the console process keeps sampling enabled cameras (RTSP and webcam on a loop; a recorded file or `MOCK` URI once per session, or again if that file changes). **Start monitoring** / **Stop monitoring** are on the alert console. `MONITOR_AUTOSTART=1` (default outside tests) starts that worker with the app. Stop is saved in `data/monitor_state.json`.

**Run** remains a one-shot pass for a lab check. It does not replace monitoring.

Humans still **review, acknowledge, dismiss, escalate, or reopen**. There is no autonomous enforcement and no facial criminal or identity labeling.

Synthetic MOCK and the Phase 3 synthetic activity demo remain **lab / development only**. They are not the customer detection path. Seeded demo cameras are still reviewed if you leave them enabled.

## Incident clips

When monitoring raises an alert, it saves a short clip from the sampled feed around that detection (default **5 seconds before and 5 seconds after**, about 10 seconds, never longer than 15). The clip is stored under `data/clips/` (`CLIP_DIR`) and plays on the alert review page. The same camera and category will not raise another alert until `MONITOR_ALERT_COOLDOWN_SEC` (default 60). Notify still runs for that alert (camera `security_email` / login email, recipients).

`MAX_INCIDENT_CLIPS` (default 400) deletes the oldest clip **files** only. Alert rows stay. There is still no UI wipe for alerts or cameras.

## Video files are for training only

Developer **Train models → Extract frames from video** samples an authorized clip into labeled JPEGs. That is not live detection. Uploads remain training-only and **developer-only**. Customer site admins and operators do not see this UI.

Run Pipeline does **not** treat a one-shot “Authorized video file upload” as the product detection path. Attached files on Run are ignored.

## Accounts

Fresh installs ship **no default password**. Create the first **site admin** with **Create account** — that first account is **auto-approved** so the product is not locked out. Later public registrations are operators and stay **pending** until a site `admin` or `developer` approves them on **Accounts**. Unapproved users cannot sign in to the console, cameras, Run Pipeline, or alerts; they are never sent a sign-in code; password reset does not bypass that gate.

After approval, customer `admin` / `operator` accounts must enter a **one-time code emailed to their login address** (`users.email`) before the console opens. Site admin uses that same gate. `security_email` remains the preferred alert-notify address and is not used for sign-in codes. If Resend is not configured, the UI says so honestly; set `AUTH_SHOW_LOGIN_CODE=1` for local demos, or `CUSTOMER_2FA_REQUIRED=0` to skip the second gate. Mun Cyber `developer` lab accounts skip email 2FA unless `DEVELOPER_2FA_REQUIRED=1`. **Sign out** ends the console cookie. A cookie from a password-only sign-in cannot skip the email code while `CUSTOMER_2FA_REQUIRED=1`. Login fields start empty; a browser may still autofill a saved site password until that saved login is removed.

Site admin and operator run cameras, alerts, My cameras, recipients, and Run Pipeline **after approval** — they **cannot** train. Optional `ADMIN_USERNAME` / `ADMIN_PASSWORD` bootstrap only if you set both yourself — never a published default. Env-seeded `ADMIN_*`, `OPERATOR_*`, and `DEVELOPER_*` accounts are active/approved when created.

Model training is reserved for the **developer** role (Mun Cyber Technologies). Seed it only on lab machines with **both** `DEVELOPER_USERNAME` and `DEVELOPER_PASSWORD` (optional `DEVELOPER_EMAIL`). Customer `.env.example` leaves these empty. Never `operator` / `changeme`. Developers can also approve customer accounts. Ordinary operators cannot.

## Retention — no UI wipe

The console does **not** hard-delete these records:

| Record | Console actions | Not offered |
|--------|-----------------|-------------|
| Alert rows | Acknowledge / Dismiss / Escalate / Reopen + audit trail | Delete / wipe the row |
| Cameras | Enable / Disable; attach or detach from My cameras / account linkage | Permanent camera delete |
| Videos | One-shot train extract; leftover files may remain on disk | Video library UI or delete library |
| Incident clips | Kept with the alert for review. Oldest files may be pruned at `MAX_INCIDENT_CLIPS` | Delete / wipe the alert or its clip from the UI |

Alerts stay in SQLite with their audit log so a review history cannot be erased from the UI. Disabled cameras remain in the registry. Train extract leftovers under `data/uploads/` (or the extract destination) are not a customer media library.

## Notify

Camera owners are notified at each linked account’s `security_email`, then login email. `RESEND_API_KEY` / `RESEND_FROM` stay empty until you configure a verified sender. Missing keys stay `queued` / `undelivered`.
