# Authentication — Mun Cyber Eye

Local operator accounts for the Flask console. **AI detects and alerts. Humans verify and decide.** Accounts do not enable autonomous enforcement.

## Sign-in options

1. **Create account** at `/register` (also `/create-account`) — the **first** account on a fresh install becomes customer **site `admin`** and is **auto-approved** so someone can approve later customers. Later users receive the `operator` role and stay **pending** until a site `admin` or `developer` approves them. Choose a strong password: **at least 12 characters, with at least one letter and one digit**. Common or default passwords (`changeme`, `password`, `password123`, `admin`, `operator`, sequential patterns like `123456`) are rejected. **No default password is shipped.** Create account **never** grants `developer` and **never** grants training.
2. **Optional customer env bootstrap** only if you set **both** `ADMIN_USERNAME` and `ADMIN_PASSWORD` in `.env` (plus optional `ADMIN_EMAIL`). If either is empty, nothing is seeded. Seeded `ADMIN_*` / `OPERATOR_*` rows are **active and approved**. `ADMIN_ROLE` may be `admin` (default) or `operator` — it cannot create a developer. Optional `OPERATOR_USERNAME` / `OPERATOR_PASSWORD` / `OPERATOR_EMAIL` is seeded the same way as an `operator` when both username and password are set. Env passwords must pass the same policy; `operator` / `changeme` is refused. Alert notify prefers each user’s optional `security_email`, then login email.
3. **Developer (Mun Cyber / lab) env seed** only if you set **both** `DEVELOPER_USERNAME` and `DEVELOPER_PASSWORD` (optional `DEVELOPER_EMAIL`). Leave these **empty on customer installs**. Mun Cyber staff set them on lab machines only. Seeded developer rows are **active and approved**. There is no published default password — never `operator` / `changeme`.

**2FA is not enforced on the Mun Cyber Eye Capstone console.** Sign-in is a password plus admin approval only. MFA on www.muncyber.com is a separate product surface and is not wired into this Flask console.

Login checks the hashed password in the `users` table **and** that the account is approved and active. Pending accounts see a clear “awaiting admin approval” message (wrong passwords still show “Invalid credentials”). Username `operator` with password `changeme` is **always** rejected, even if an old database still has that hash. Password reset is not issued for pending or inactive accounts and cannot grant console access while pending. After any optional seed, the database is the source of truth. Session keys remain `user` (username) and `role`.

### Password policy

| Rule | Detail |
|------|--------|
| Minimum length | 12 characters |
| Complexity | At least one letter and one digit. Symbols are allowed, not required. |
| Rejected | `changeme`, `password`, `password123`, `admin`, `operator`, and obvious sequences (`123456`, `qwerty`, `abcdef`) |
| Applied on | Create account, password reset, env seed, and every `set_password` path |

### Legacy Capstone `operator` / `changeme`

`.env.example` does **not** seed those credentials. On boot, `DISABLE_LEGACY_OPERATOR=1` (product default) deactivates an existing `operator` row **only if** the password still matches `changeme` **or** the email is the legacy seed `operator@localhost`. A customer who chose username `operator` with a strong password and a different email is left active. You can also run `python scripts/disable_legacy_operator.py` (documented in [STANDALONE.md](STANDALONE.md)). If that row was your only admin, seed a new site admin with `ADMIN_USERNAME` / `ADMIN_PASSWORD` (a strong password you choose) and restart.

## Pages

| Path | Purpose |
|------|---------|
| `/login` | Username / password, forgot-password and create-account links. Empty user table points at first site-admin setup. |
| `/register` | Username, email, password, confirm password. First account is site admin (not developer) and is auto-approved. Later accounts wait for approval. Password policy is shown on the form. |
| `/forgot-password` | Request a time-limited reset (45 minutes by default). Pending accounts are treated like unknown identities. |
| `/reset-password?token=…` | Set a new password (approved, active accounts only) |
| `/logout` | Clear the session |
| `/accounts` | Admin dashboard: pending queue with **Approve** / **Reject**, deactivate / reactivate. Site `admin` and `developer` only — not `operator`. |
| `/admin/train` (`/train`) | Model training — **developer only**; customers do not see the nav or the page |

## Password reset and email

Optional Resend delivery uses `RESEND_API_KEY` and `RESEND_FROM` (`alerts/notify.py`, stdlib `urllib` with an explicit User-Agent).

- If Resend **is** configured and the provider accepts the message, the page shows a generic success line and does **not** say whether the account exists.
- If Resend is **not** configured, the page says email delivery is not configured. It never claims “email sent.”
- A one-time reset URL is always written to the application log when a matching **approved and active** account exists. Set `AUTH_SHOW_RESET_URL=1` to also print that URL on the page for local demos.
- Resetting a password does **not** bypass approval. Pending accounts cannot use the console until approved.

## Approval

Customers register → wait for approval → then use cameras / alerts / Run Pipeline. Training stays developer-only.

| Who | Created how | Sign-in |
|-----|-------------|---------|
| First Create account | Public register on empty `users` table | Auto-approved site `admin` |
| Later Create account | Public register | **Pending** until a site `admin` or `developer` approves |
| `ADMIN_*` / `OPERATOR_*` / `DEVELOPER_*` | Env seed when username **and** password are set | Active and approved when created |
| Ordinary `operator` | — | Cannot approve or reject anyone |

Approve / reject / deactivate writes `system_audit` (`approve_account`, `reject_account`, `deactivate_account`, `reactivate_account`) with the acting username. Disabling a leftover Capstone `operator` seed writes `disable_legacy_operator` (actor `startup` or `script`). Existing auth databases are grandfathered as approved (`approved=1`).

## Data

`users` lives in `AUTH_DB_PATH` (default `data/auth.db`), separate from the Phase 4-style operators email directory and from `alerts`. Fields: `id`, `username`, `email`, `password_hash` (Werkzeug), `role` (`admin` \| `operator` \| `developer`), `active`, `approved`, `approved_at`, `approved_by`, `created_at`, `reset_token`, `reset_expires`, `security_email` (optional alert address).

Register collects login email and optional security email. Attach cameras on **My cameras** / **Accounts**. A linked account with no security or login email is skipped on notify — see [CAMERA_OWNER_NOTIFY.md](CAMERA_OWNER_NOTIFY.md).

Optional site fallback (not a personal inbox): `SECURITY_ALERT_EMAIL` (default empty). Used only when no camera account and no signed-in user apply.

## Safety

## Roles and model training

| Role | How it is created | Cameras / alerts / recipients | Approve customers | Train / activate / upload / extract |
|------|-------------------|-------------------------------|-------------------|-------------------------------------|
| `admin` | First Create account (auto-approved), or `ADMIN_USERNAME` when both username and password are set | Yes after approval (site ops) | **Yes** | No |
| `operator` | Later Create account (pending), or `OPERATOR_*` when both are set | Yes after approval | No | No |
| `developer` | `DEVELOPER_USERNAME` + `DEVELOPER_PASSWORD` on lab machines only | Lab console (cameras / review) | **Yes** | **Yes** |

Customers use **shipped checkpoints**. Training is Mun Cyber developer-side. Public Create account never creates `developer`. `ADMIN_ROLE` cannot be used to grant training.

There is no promote UI in this prototype. Do not grant training by flipping `users.role` to `admin` — that role is site ops only.

See [ADMIN_TRAINING.md](ADMIN_TRAINING.md).

Authorized use only. Registration creates a human reviewer, not an automated responder.
