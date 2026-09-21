# Product operations and retention — Mun Cyber Eye

**AI detects and alerts. Humans verify and decide.**

This note records owner product rules for Mun Cyber Technologies. See also [ETHICS_AND_SAFETY.md](ETHICS_AND_SAFETY.md) and [STANDALONE.md](STANDALONE.md).

## Detection path

Elevated-risk / violent-conduct detection is **automation from authorized cameras** in the registry:

- File-source cameras
- RTSP cameras
- Webcam only if `ALLOW_WEBCAM=1`

The pipeline runs those feeds (one camera or all enabled, sequentially in this prototype). Humans still **review, acknowledge, dismiss, escalate, or reopen**. There is no autonomous enforcement and no facial criminal or identity labeling.

Synthetic MOCK and the Phase 3 synthetic activity demo remain **lab / development only**. They are not the customer detection path.

## Video files are for training only

Developer **Train models → Extract frames from video** samples an authorized clip into labeled JPEGs. That is not live detection. Uploads remain training-only and **developer-only**. Customer site admins and operators do not see this UI.

Run Pipeline does **not** treat a one-shot “Authorized video file upload” as the product detection path. Attached files on Run are ignored.

## Accounts

Fresh installs ship **no default password**. Create the first **site admin** with **Create account**. Later registrations are operators. Site admin and operator run cameras, alerts, My cameras, recipients, and Run Pipeline — they **cannot** train. Optional `ADMIN_USERNAME` / `ADMIN_PASSWORD` bootstrap only if you set both yourself — never a published default.

Model training is reserved for the **developer** role (Mun Cyber Technologies). Seed it only on lab machines with **both** `DEVELOPER_USERNAME` and `DEVELOPER_PASSWORD` (optional `DEVELOPER_EMAIL`). Customer `.env.example` leaves these empty. Never `operator` / `changeme`.

## Retention — no UI wipe

The console does **not** hard-delete these records:

| Record | Console actions | Not offered |
|--------|-----------------|-------------|
| Alert rows | Acknowledge / Dismiss / Escalate / Reopen + audit trail | Delete / wipe the row |
| Cameras | Enable / Disable; attach or detach from My cameras / account linkage | Permanent camera delete |
| Videos | One-shot train extract; leftover files may remain on disk | Video library UI or delete library |

Alerts stay in SQLite with their audit log so a review history cannot be erased from the UI. Disabled cameras remain in the registry. Train extract leftovers under `data/uploads/` (or the extract destination) are not a customer media library.

## Notify

Camera owners are notified at each linked account’s `security_email`, then login email. `RESEND_API_KEY` / `RESEND_FROM` stay empty until you configure a verified sender. Missing keys stay `queued` / `undelivered`.
