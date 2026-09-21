# Camera-account notification

**AI detects and alerts. Humans verify and decide.** Email is notify-only. It does not close a review or act on anyone.

Many cameras can belong to **one account**, and one camera can notify **several accounts**. Assignments are edited on **My cameras** (every signed-in user) and admin **Accounts**, not only on the camera form.

When a pipeline run creates an alert for camera C, outbound Resend mail is the **union** of:

1. **Every account linked to C** — each user’s `security_email` if set, otherwise their login `email`. Inactive, **pending (unapproved)**, and users with no address are skipped with an honest note.
2. **Signed-in account** — on interactive Flask **Run Pipeline** / resend only (`security_email` or login email). Background jobs do not invent a session user.
3. **Optional extras** — Recipients directory, `ALERT_EMAIL_RECIPIENTS` when non-empty, and optional per-camera `notify_email`.
4. **`SECURITY_ALERT_EMAIL`** — site-wide fallback **only** when no camera account and no session user contributed an address. Default empty.

Addresses are lowercased and de-duplicated. **Never** bake a personal inbox (for example a Gmail address) into defaults, seeds, or docs.

## Assigning cameras on an account

1. Sign in → **My cameras**.
2. Optionally set **Security email** (preferred for alerts).
3. Check or uncheck cameras. Save.

Admins can do the same for any user at **Accounts**. The camera Add/Edit form still shows linked accounts as a checklist.

Seeded demo cameras start with **no** linked accounts.

## Auth emails

| Field | Purpose |
|-------|---------|
| `email` | Login / password reset |
| `security_email` | Optional; preferred for alert notify |
| `ADMIN_EMAIL` | Seeded admin login email (prototype default `{ADMIN_USERNAME}@localhost`) |
| `SECURITY_ALERT_EMAIL` | Optional site fallback; default empty |

Register collects login email and optional security email.

## Honest delivery

- A missing `RESEND_API_KEY` is `queued` / `undelivered` — never `sent`.
- A linked account with no security or login email is skipped; other recipients still receive the attempt.
- API keys are never written to logs.

## Email copy

Subject and body name **severity**, **category**, **camera**, and **place**. The attempted list is stored as `metadata.notified_emails` and in `delivery_log`.

See also [PHASE4.md](PHASE4.md) and [AUTH.md](AUTH.md).
