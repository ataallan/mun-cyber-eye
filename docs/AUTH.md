# Authentication — Mun Cyber Eye

Local operator accounts for the Flask console. **AI detects and alerts. Humans verify and decide.** Accounts do not enable autonomous enforcement.

## Sign-in options

1. **Create account** at `/register` (also `/create-account`) — the **first** account on a fresh install becomes customer **site `admin`**. Later users receive the `operator` role. Choose a strong password (minimum 8 characters). **No default password is shipped.** Create account **never** grants `developer` and **never** grants training.
2. **Optional customer env bootstrap** only if you set **both** `ADMIN_USERNAME` and `ADMIN_PASSWORD` in `.env` (plus optional `ADMIN_EMAIL`). If either is empty, nothing is seeded. `ADMIN_ROLE` may be `admin` (default) or `operator` — it cannot create a developer. Optional `OPERATOR_USERNAME` / `OPERATOR_PASSWORD` / `OPERATOR_EMAIL` is seeded the same way as an `operator` when both username and password are set. Alert notify prefers each user’s optional `security_email`, then login email.
3. **Developer (Mun Cyber / lab) env seed** only if you set **both** `DEVELOPER_USERNAME` and `DEVELOPER_PASSWORD` (optional `DEVELOPER_EMAIL`). Leave these **empty on customer installs**. Mun Cyber staff set them on lab machines only. There is no published default password — never `operator` / `changeme`.

Login checks the hashed password in the `users` table. After any optional seed, the database is the source of truth. Session keys remain `user` (username) and `role`.

## Pages

| Path | Purpose |
|------|---------|
| `/login` | Username / password, forgot-password and create-account links. Empty user table points at first site-admin setup. |
| `/register` | Username, email, password, confirm password. First account is site admin (not developer). |
| `/forgot-password` | Request a time-limited reset (45 minutes by default) |
| `/reset-password?token=…` | Set a new password |
| `/logout` | Clear the session |
| `/admin/train` (`/train`) | Model training — **developer only**; customers do not see the nav or the page |

## Password reset and email

Optional Resend delivery uses `RESEND_API_KEY` and `RESEND_FROM` (`alerts/notify.py`, stdlib `urllib` with an explicit User-Agent).

- If Resend **is** configured and the provider accepts the message, the page shows a generic success line and does **not** say whether the account exists.
- If Resend is **not** configured, the page says email delivery is not configured. It never claims “email sent.”
- A one-time reset URL is always written to the application log when a matching active account exists. Set `AUTH_SHOW_RESET_URL=1` to also print that URL on the page for local demos.

## Data

`users` lives in `AUTH_DB_PATH` (default `data/auth.db`), separate from the Phase 4-style operators email directory and from `alerts`. Fields: `id`, `username`, `email`, `password_hash` (Werkzeug), `role` (`admin` \| `operator` \| `developer`), `active`, `created_at`, `reset_token`, `reset_expires`, `security_email` (optional alert address).

Register collects login email and optional security email. Attach cameras on **My cameras** / **Accounts**. A linked account with no security or login email is skipped on notify — see [CAMERA_OWNER_NOTIFY.md](CAMERA_OWNER_NOTIFY.md).

Optional site fallback (not a personal inbox): `SECURITY_ALERT_EMAIL` (default empty). Used only when no camera account and no signed-in user apply.

## Safety

## Roles and model training

| Role | How it is created | Cameras / alerts / recipients | Train / activate / upload / extract |
|------|-------------------|-------------------------------|-------------------------------------|
| `admin` | First Create account, or `ADMIN_USERNAME` when both username and password are set | Yes (site ops) | No |
| `operator` | Later Create account, or `OPERATOR_*` when both are set | Yes | No |
| `developer` | `DEVELOPER_USERNAME` + `DEVELOPER_PASSWORD` on lab machines only | Lab console (cameras / review) | **Yes** |

Customers use **shipped checkpoints**. Training is Mun Cyber developer-side. Public Create account never creates `developer`. `ADMIN_ROLE` cannot be used to grant training.

There is no promote UI in this prototype. Do not grant training by flipping `users.role` to `admin` — that role is site ops only.

See [ADMIN_TRAINING.md](ADMIN_TRAINING.md).

Authorized use only. Registration creates a human reviewer, not an automated responder.
