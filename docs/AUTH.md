# Authentication — Mun Cyber Eye

Local operator accounts for the Flask console. **AI detects and alerts. Humans verify and decide.** Accounts do not enable autonomous enforcement.

## Sign-in options

1. **Create account** at `/register` (also `/create-account`) — the **first** account on a fresh install becomes customer **site `admin`** and is **auto-approved** so someone can approve later customers. Later users receive the `operator` role and stay **pending** until a site `admin` or `developer` approves them. Choose a strong password (minimum 8 characters). **No default password is shipped.** Create account **never** grants `developer` and **never** grants training.
2. **Optional customer env bootstrap** only if you set **both** `ADMIN_USERNAME` and `ADMIN_PASSWORD` in `.env` (plus optional `ADMIN_EMAIL`). If either is empty, nothing is seeded. Seeded `ADMIN_*` / `OPERATOR_*` rows are **active and approved**. `ADMIN_ROLE` may be `admin` (default) or `operator` — it cannot create a developer. Optional `OPERATOR_USERNAME` / `OPERATOR_PASSWORD` / `OPERATOR_EMAIL` is seeded the same way as an `operator` when both username and password are set. Alert notify prefers each user’s optional `security_email`, then login email.
3. **Developer (Mun Cyber / lab) env seed** only if you set **both** `DEVELOPER_USERNAME` and `DEVELOPER_PASSWORD` (optional `DEVELOPER_EMAIL`). Leave these **empty on customer installs**. Mun Cyber staff set them on lab machines only. Seeded developer rows are **active and approved**. There is no published default password — never `operator` / `changeme`.

Login checks the hashed password in the `users` table **and** that the account is approved and active. Pending accounts see a clear “awaiting admin approval” message (wrong passwords still show “Invalid credentials”). Password reset is not issued for pending or inactive accounts and cannot grant console access while pending. After any optional seed, the database is the source of truth. Session keys are `user` (username), `role`, and — only after a successful email code — `email_2fa_ok` plus `email_2fa_at`.

**Second gate (customers only):** after password + approval succeed, customer `admin` / `operator` accounts must enter a **one-time email code** before cameras, alerts, or Run Pipeline. Site admin and approved operators use the same gate. Pending users are never sent a code. Mun Cyber `developer` lab accounts skip this step unless `DEVELOPER_2FA_REQUIRED=1`. A cookie that never completed the code (including one saved before email codes were required) cannot open the console while `CUSTOMER_2FA_REQUIRED=1`.

## Pages

| Path | Purpose |
|------|---------|
| `/login` | Username / password, forgot-password and create-account links. Empty user table points at first site-admin setup. Approved customers then enter an email code (`CUSTOMER_2FA_REQUIRED=1`). |
| `/login-code` | One-time sign-in code after password + approval. Not shown to pending users. Submits when all 6 digits are entered; the verify button remains as a fallback. |
| `/register` | Username, email, password, confirm password. First account is site admin (not developer) and is auto-approved. Later accounts wait for approval. |
| `/forgot-password` | Request a time-limited reset (45 minutes by default). Pending accounts are treated like unknown identities. |
| `/reset-password?token=…` | Set a new password (approved, active accounts only) |
| `/logout` | **Sign out.** Clears the session cookie, including any email-code stamp, so the next sign-in starts at the password form |
| `/accounts` | Admin dashboard: pending queue with **Approve** / **Reject**, deactivate / reactivate. Site `admin` and `developer` only — not `operator`. |
| `/admin/train` (`/train`) | Model training — **developer only**; customers do not see the nav or the page |

## Password reset and email

Optional Resend delivery uses `RESEND_API_KEY` and `RESEND_FROM` (`alerts/notify.py`, stdlib `urllib` with an explicit User-Agent).

- If Resend **is** configured and the provider accepts the message, the page shows a generic success line and does **not** say whether the account exists.
- If Resend is **not** configured, the page says email delivery is not configured. It never claims “email sent.”
- A one-time reset URL is always written to the application log when a matching **approved and active** account exists. Set `AUTH_SHOW_RESET_URL=1` to also print that URL on the page for local demos.
- Resetting a password does **not** bypass approval. Pending accounts cannot use the console until approved.

## Email sign-in codes (customer 2FA)

Approval is the first gate. An emailed one-time code is the second gate for **customer** `admin` / `operator` accounts (including the first auto-approved site admin). This is **not** an authenticator-app / TOTP enrollment flow.

| Who | After password succeeds |
|-----|-------------------------|
| Pending (unapproved) | Blocked. **No code is created or emailed.** |
| Approved customer `admin` / `operator` | Code hashed at rest, 10-minute TTL, single-use, rate-limited. Sent to **login email** (`users.email` from Create account / env seed) — not `security_email` (that address is for alert notify). |
| First Create account site admin | Same email-code step on login (`CUSTOMER_2FA_REQUIRED=1`). |
| `developer` (Mun Cyber lab) | **Not required** unless `DEVELOPER_2FA_REQUIRED=1`. |

Delivery uses the same Resend helper as password reset (`alerts/notify.py`):

- If Resend **is** configured and the provider accepts the message, the verify page says a code was sent to the login email. It does not invent success when the provider fails.
- If Resend is **not** configured, the page says email delivery is not configured and **never claims the code was emailed**.
- A one-time code is always written to the application log when issued for an approved, active account. Set `AUTH_SHOW_LOGIN_CODE=1` to also print that code on the verify page for local Capstone demos (mirrors `AUTH_SHOW_RESET_URL`).
- Set `CUSTOMER_2FA_REQUIRED=0` to skip the email-code step for local demos that only need password + approval.

Codes are stored as Werkzeug password hashes (`login_code_hash`), not plaintext. Failed attempts are limited; requesting a new code is throttled (`LOGIN_CODE_RESEND_SECONDS`, default 45).

The verify page keeps a **Verify and open console** button. With JavaScript, the form also submits once the field holds `LOGIN_CODE_DIGITS` (6) digits — including paste and mobile one-time-code autofill (`inputmode="numeric"`, `autocomplete="one-time-code"`, `maxlength="6"`). A wrong code is rejected on the server, the field is shown empty again, and the person can enter another code. The page does not resubmit a failed code by itself.

## Session, Sign out, and saved browser passwords

Site `admin` and approved `operator` accounts share the email-code gate. Completing `/login-code` sets `email_2fa_ok` and a timestamp. Password-only sign-in (developer skip, or `CUSTOMER_2FA_REQUIRED=0`) does **not** set that stamp. While customer email codes are required, `login_required` rejects a session that lacks it: the cookie is cleared and the next request is the login form. That closes the hole where a session started before email codes were live, or created with password alone, skipped `/login-code` until someone signed out.

**Sign out** (`/logout`, labeled Sign out in the header) ends that held session immediately. Cookies are browser session cookies (not a “remember me” permanent cookie). `SESSION_HOURS` (default 8) also rejects a cookie that stays open longer than that, and `PERMANENT_SESSION_LIFETIME` uses the same cap if a session is ever marked permanent.

The login form renders username and password **empty**. It sets `autocomplete="off"` on the form and both fields, and the fields stay read-only until focused, so they are not pre-filled from the server and browsers are asked not to drop a saved password into them. Chrome and other password managers can still refill a saved login for this site (for example `http://127.0.0.1:5055`) until the user removes that saved password. The server never writes the password into the HTML.

## Approval

Customers register → wait for approval → then complete an email sign-in code → then use cameras / alerts / Run Pipeline. Training stays developer-only. Pending accounts have neither console access nor 2FA.

| Who | Created how | Sign-in |
|-----|-------------|---------|
| First Create account | Public register on empty `users` table | Auto-approved site `admin`, then email sign-in code |
| Later Create account | Public register | **Pending** until a site `admin` or `developer` approves; then email sign-in code |
| `ADMIN_*` / `OPERATOR_*` | Env seed when username **and** password are set | Active and approved when created; then email sign-in code |
| `DEVELOPER_*` | Env seed on lab machines | Active and approved; email 2FA off by default |
| Ordinary `operator` | — | Cannot approve or reject anyone |

Approve / reject / deactivate writes `system_audit` (`approve_account`, `reject_account`, `deactivate_account`, `reactivate_account`) with the acting username. Existing auth databases are grandfathered as approved (`approved=1`).

## Data

`users` lives in `AUTH_DB_PATH` (default `data/auth.db`), separate from the Phase 4-style operators email directory and from `alerts`. Fields: `id`, `username`, `email`, `password_hash` (Werkzeug), `role` (`admin` \| `operator` \| `developer`), `active`, `approved`, `approved_at`, `approved_by`, `created_at`, `reset_token`, `reset_expires`, `security_email` (optional alert address), `login_code_hash` / `login_code_expires` / `login_code_created_at` / `login_code_attempts` (ephemeral email sign-in codes).

Register collects login email and optional security email. Attach cameras on **My cameras** / **Accounts**. A linked account with no security or login email is skipped on notify — see [CAMERA_OWNER_NOTIFY.md](CAMERA_OWNER_NOTIFY.md).

Optional site fallback (not a personal inbox): `SECURITY_ALERT_EMAIL` (default empty). Used only when no camera account and no signed-in user apply.

## Safety

## Roles and model training

| Role | How it is created | Cameras / alerts / recipients | Approve customers | Train / activate / upload / extract |
|------|-------------------|-------------------------------|-------------------|-------------------------------------|
| `admin` | First Create account (auto-approved), or `ADMIN_USERNAME` when both username and password are set | Yes after approval **and** email sign-in code | **Yes** | No |
| `operator` | Later Create account (pending), or `OPERATOR_*` when both are set | Yes after approval **and** email sign-in code | No | No |
| `developer` | `DEVELOPER_USERNAME` + `DEVELOPER_PASSWORD` on lab machines only | Lab console (cameras / review); email 2FA off by default | **Yes** | **Yes** |

Customers use **shipped checkpoints**. Training is Mun Cyber developer-side. Public Create account never creates `developer`. `ADMIN_ROLE` cannot be used to grant training.

There is no promote UI in this prototype. Do not grant training by flipping `users.role` to `admin` — that role is site ops only.

See [ADMIN_TRAINING.md](ADMIN_TRAINING.md).

Authorized use only. Registration creates a human reviewer, not an automated responder.
