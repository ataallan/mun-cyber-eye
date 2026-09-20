# Phase 4 — Alert System

**Mun Cyber Eye** extends the Phase 2/3 human-in-the-loop console with a richer structured alert schema and pluggable, honest outbound delivery.

**AI detects and alerts. Humans verify and decide.**

Outbound email and webhooks notify authorized personnel. They do **not** lock doors, dispatch force, or close a review. The console remains the primary surface.

## What shipped

| Piece | Location |
|-------|----------|
| Structured alert fields | `alerts/store.py` (`Alert`) + `alerts/schema.py` |
| Resend email adapter | `alerts/notify.py` (`EmailAdapter`) |
| SIEM/SOAR webhook adapter | `alerts/notify.py` (`WebhookAdapter`) |
| Delivery + retry log | `delivery_log` table, `NotificationService` |
| Authorized recipients | `.env` `ALERT_EMAIL_RECIPIENTS` + `operators` table |
| Alert detail + resend | `app/templates/alert_detail.html` |
| Recipient management UI | `/recipients` (admin / operator) |
| Tests | `tests/test_alert_schema.py`, `tests/test_notify.py`, `tests/test_delivery.py`, `tests/test_app_phase4.py` |

Phase 2 review actions (acknowledge / dismiss / escalate / reopen) and the Phase 3 activity path are unchanged. Delivery attempts are written to the same `audit_log` so they sit next to human decisions.

## Structured payload

Every alert now carries responder-facing fields (existing Phase 2 columns are kept):

| Field | Meaning |
|-------|---------|
| `severity` | `info` / `warning` / `critical` (from `risk_level`) |
| `category` | Phase 2/3 activity category |
| `confidence` | Model / heuristic score 0–1 |
| `location_label` | Site label (`DEFAULT_LOCATION_LABEL`) |
| `camera_id` | Source id (`DEFAULT_CAMERA_ID`) |
| `source` | Authorized camera label |
| `frame_time` | `HH:MM:SS.ss` from `timestamp_sec` |
| `snapshot_path` | JPEG attached to the alert |
| `short_rationale` | First sentence of the rationale |
| `recommended_human_action` | Advisory SOP hint only |
| `correlation_id` | Shared id for alerts from the same pipeline run |
| `created_at` | UTC timestamp |
| `delivery_status` | Outbound channel state (see below) |
| `human_status` | `open` / `acknowledged` / `dismissed` / `escalated` |

JSON for an alert (session auth required):

```
GET /alerts/<id>.json
```

The payload always includes:

```
safety: "AI detects and alerts. Humans verify and decide."
enforcement: "none — advisory alert only; no autonomous enforcement"
```

## Delivery statuses

| Status | Meaning |
|--------|---------|
| `pending` | Created, outbound not attempted |
| `queued` | Channel intended but cannot send (typical: no `RESEND_API_KEY`) |
| `sent` | Every configured outbound channel accepted the message |
| `partial` | At least one channel succeeded and another failed or stayed queued |
| `failed` | Configured channels were attempted and none succeeded |
| `undelivered` | No outbound channel was configured |

**Never invented:** a missing Resend key is `queued` / `undelivered`, not `sent`. MOCK / offline mode still generates alerts in the console.

Retries: each channel is tried up to `ALERT_NOTIFY_MAX_ATTEMPTS` (default 3) on transport failure. Missing configuration is not retried.

## Recipients

Authorized emails come from two sources and are merged (unique, lowercased):

1. `ALERT_EMAIL_RECIPIENTS` — comma-separated list in `.env`
2. `operators` table — managed at **Recipients** by `admin` or `operator` sessions

Inactive directory rows are skipped. Environment recipients cannot be deleted from the UI.

Roles:

| Login | Role | Recipients + resend |
|-------|------|---------------------|
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | `admin` (default) | Yes |
| `OPERATOR_USERNAME` / `OPERATOR_PASSWORD` | `operator` | Yes |

Unauthenticated users cannot manage recipients.

## Demo: email delivery with Resend

1. Create an API key at [resend.com](https://resend.com) (sending access is enough).
2. Verify `muncyber.com` (or another domain you control) so the From address is accepted. Default From is `Mun Cyber Technologies <info@muncyber.com>`.
3. Copy `.env.example` to `.env` and set:

```bash
RESEND_API_KEY=re_xxxxxxxx
RESEND_FROM=Mun Cyber Technologies <info@muncyber.com>
ALERT_EMAIL_RECIPIENTS=you@your-domain.com
```

   Or sign in and add the same address on **Recipients**.

4. Start the console (`python run.py`), sign in, **Run Pipeline → Synthetic demo**.
5. Open an alert. Delivery should be `sent` (or `partial` if only email is configured and succeeds). The inbox should show the structured message with the safety banner.
6. Use **Resend to authorized recipients** to record another attempt. The delivery log shows HTTP success or the provider error body.

Without a key, the same walkthrough still works: delivery stays `queued` with `RESEND_API_KEY not configured; email not sent`. That is intentional.

The Resend adapter uses stdlib `urllib` and always sends a `User-Agent` (`MunCyberEye/4.0 (+https://muncyber.com)`). Resend sits behind Cloudflare; a missing User-Agent is often blocked as error 1010.

## Demo: webhook (SIEM/SOAR)

```bash
ALERT_WEBHOOK_URL=https://example.invalid/soar/alerts
ALERT_WEBHOOK_SECRET=optional-bearer-token
```

The body is:

```json
{
  "event": "mun_cyber_eye.alert",
  "safety": "AI detects and alerts. Humans verify and decide.",
  "alert": { "...structured payload..." }
}
```

If `ALERT_WEBHOOK_SECRET` is set, the request includes `Authorization: Bearer <secret>`.

## Honest offline / MOCK mode

- No GPU, no YOLO, no Resend key: pipeline + console still run.
- Alerts are stored and reviewable.
- Outbound status is `queued` or `undelivered`, never a fake success.
- Tests mock `urlopen`; they do not call Resend.

## Non-goals (Phase 4)

- Autonomous lockdown, weapons discharge, or emergency-service auto-dial
- Public or unverified recipient lists
- Guaranteed email delivery (we report the provider’s result)
- Replacing human acknowledge / dismiss / escalate
