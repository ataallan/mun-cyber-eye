# Authentication — Mun Cyber Eye

Local operator accounts for the Flask console. **AI detects and alerts. Humans verify and decide.** Accounts do not enable autonomous enforcement.

## Sign-in options

1. **Create account** at `/register` (also `/create-account`) — new users receive the `operator` role.
2. **Bootstrap admin** from `.env`: `ADMIN_USERNAME` / `ADMIN_PASSWORD` (defaults `operator` / `changeme`). On first boot the app inserts that user into SQLite (or syncs the password when `ADMIN_SYNC_PASSWORD=1`, the prototype default). Optional `OPERATOR_USERNAME` / `OPERATOR_PASSWORD` is seeded the same way as an `operator`.

Login checks the hashed password in the `users` table. After seed, the database is the source of truth. Session keys remain `user` (username) and `role`.

## Pages

| Path | Purpose |
|------|---------|
| `/login` | Username / password, forgot-password and create-account links |
| `/register` | Username, email, password, confirm password |
| `/forgot-password` | Request a time-limited reset (45 minutes by default) |
| `/reset-password?token=…` | Set a new password |
| `/logout` | Clear the session |

## Password reset and email

Optional Resend delivery uses `RESEND_API_KEY` and `RESEND_FROM` (`alerts/notify.py`, stdlib `urllib` with an explicit User-Agent).

- If Resend **is** configured and the provider accepts the message, the page shows a generic success line and does **not** say whether the account exists.
- If Resend is **not** configured, the page says email delivery is not configured. It never claims “email sent.”
- A one-time reset URL is always written to the application log when a matching active account exists. Set `AUTH_SHOW_RESET_URL=1` to also print that URL on the page for local demos.

## Data

`users` lives in `AUTH_DB_PATH` (default `data/auth.db`), separate from the Phase 4-style operators email directory and from `alerts`. Fields: `id`, `username`, `email`, `password_hash` (Werkzeug), `role` (`admin` \| `operator`), `active`, `created_at`, `reset_token`, `reset_expires`.

## Safety

Authorized use only. Registration creates a human reviewer, not an automated responder.
