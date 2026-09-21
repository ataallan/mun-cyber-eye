# Authentication — Mun Cyber Eye

Local operator accounts for the Flask console. **AI detects and alerts. Humans verify and decide.** Accounts do not enable autonomous enforcement.

## Sign-in options

1. **Create account** at `/register` (also `/create-account`) — the **first** account on a fresh install becomes `admin`. Later users receive the `operator` role. Choose a strong password (minimum 8 characters). **No default password is shipped.**
2. **Optional env bootstrap** only if you set **both** `ADMIN_USERNAME` and `ADMIN_PASSWORD` in `.env` (plus optional `ADMIN_EMAIL`). If either is empty, nothing is seeded. Optional `OPERATOR_USERNAME` / `OPERATOR_PASSWORD` / `OPERATOR_EMAIL` is seeded the same way as an `operator` when both username and password are set. Alert notify prefers each user’s optional `security_email`, then login email.

Login checks the hashed password in the `users` table. After any optional seed, the database is the source of truth. Session keys remain `user` (username) and `role`.

## Pages

| Path | Purpose |
|------|---------|
| `/login` | Username / password, forgot-password and create-account links. Empty user table points at first-admin setup. |
| `/register` | Username, email, password, confirm password. First account is admin. |
| `/forgot-password` | Request a time-limited reset (45 minutes by default) |
| `/reset-password?token=…` | Set a new password |
| `/logout` | Clear the session |
| `/admin/train` (`/train`) | Model training — **admin** trains / activates / uploads; operators may view status |

## Password reset and email

Optional Resend delivery uses `RESEND_API_KEY` and `RESEND_FROM` (`alerts/notify.py`, stdlib `urllib` with an explicit User-Agent).

- If Resend **is** configured and the provider accepts the message, the page shows a generic success line and does **not** say whether the account exists.
- If Resend is **not** configured, the page says email delivery is not configured. It never claims “email sent.”
- A one-time reset URL is always written to the application log when a matching active account exists. Set `AUTH_SHOW_RESET_URL=1` to also print that URL on the page for local demos.

## Data

`users` lives in `AUTH_DB_PATH` (default `data/auth.db`), separate from the Phase 4-style operators email directory and from `alerts`. Fields: `id`, `username`, `email`, `password_hash` (Werkzeug), `role` (`admin` \| `operator`), `active`, `created_at`, `reset_token`, `reset_expires`, `security_email` (optional alert address).

Register collects login email and optional security email. Attach cameras on **My cameras** / **Accounts**. A linked account with no security or login email is skipped on notify — see [CAMERA_OWNER_NOTIFY.md](CAMERA_OWNER_NOTIFY.md).

Optional site fallback (not a personal inbox): `SECURITY_ALERT_EMAIL` (default empty). Used only when no camera account and no signed-in user apply.

## Safety

## Roles and model training

The first Create account is `admin`. Later registered users are `operator`. Only `role == admin` can train a checkpoint, activate it, or upload labeled frames / extract video. Optional env `ADMIN_USERNAME` / `ADMIN_PASSWORD` is admin when both are set and `ADMIN_ROLE=admin` (the default). There is no promote UI; change `users.role` in `AUTH_DB_PATH` to grant training.

See [ADMIN_TRAINING.md](ADMIN_TRAINING.md).

Authorized use only. Registration creates a human reviewer, not an automated responder.
