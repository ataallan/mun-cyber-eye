# Standalone Windows download — Mun Cyber Eye

Mun Cyber Eye ships as a **Chrome-style local console**: unzip, install Python packages into `.venv`, start `run.py`, and open a browser to the login / create-account page. Detection is local to authorized cameras on this machine. There is no bundled password and no bundled Resend secret.

**AI detects and alerts. Humans verify and decide.**

## Product rules (shipped)

- **Detection** runs from **registered authorized cameras** (RTSP, file-source cameras, webcam if enabled) when the pipeline runs. Humans review, acknowledge, dismiss, escalate, or reopen. No autonomous enforcement. No facial criminal or identity labeling.
- **Video file uploads are for training only** (Mun Cyber developer → Train models → Extract frames from video). Run Pipeline is not a customer “upload an incident clip for live detection” path. Customer installs use shipped checkpoints; training is developer-side.
- **No default credentials.** Create the first site admin on **Create account** with a password you choose. That first account is auto-approved and can approve later customers. On sign-in it confirms a one-time code sent to its login email (`CUSTOMER_2FA_REQUIRED=1`, product default). It cannot train models. Later Create account users wait for admin approval — they have neither console access nor a sign-in code until approved. Set `CUSTOMER_2FA_REQUIRED=0` only for local demos that skip the email-code step. Set `AUTH_SHOW_LOGIN_CODE=1` to print the code on the verify page when Resend is not configured (the UI will not claim a message was emailed).
- **No delete wipe** for alert rows, cameras, or a video library. See [PRODUCT_OPS.md](PRODUCT_OPS.md).

Synthetic MOCK on Run Pipeline is **lab / development only**.

## Requirements

- Windows 10 or 11
- [Python 3.10+](https://www.python.org/downloads/) with **Add python.exe to PATH**
- Unzipped copy of this product (or `mun-cyber-eye-standalone.zip`)

## Install and run

1. Unzip the folder.
2. Double-click **`install_and_run.bat`** (or right-click **`install_and_run.ps1`** → Run with PowerShell).
3. Confirm the prompt: packages will be installed from `requirements.txt` into a local `.venv`.
4. The console starts at **http://127.0.0.1:5055** and the browser opens to login / create-account.
5. Create the first site admin account (cameras, alerts, recipients). It is auto-approved. Sign in with the password you chose, then enter the email sign-in code (or the on-page demo code if `AUTH_SHOW_LOGIN_CODE=1` and Resend is not configured). Later Create account users are operators and stay pending until you approve them on **Accounts**. Neither role can train models.

If `.env` is missing, the launcher copies `.env.example`. That file has **empty** `RESEND_API_KEY` / `RESEND_FROM` placeholders. Camera owners get alerts via each account’s `security_email`, then login email — you do not put secrets in the zip.

Do not commit a real `.env`. Do not ship `.venv`.

### Manual equivalent

```powershell
cd path\to\mun-cyber-eye
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
$env:MUN_OPEN_BROWSER = "1"
python run.py
```

## After sign-in

1. **Accounts** — approve pending customer registrations (site admin or lab developer). Until approved, those users cannot sign in and are never sent a sign-in code. After approval they confirm an email code, then use cameras / alerts.
2. **Cameras** — confirm or add authorized RTSP / file sources (webcam only if you set `ALLOW_WEBCAM=1`).
3. **Run Pipeline** — registered cameras (product path). MOCK is lab-only.
4. **Alerts** — acknowledge / dismiss / escalate / reopen. Rows are retained.
5. Customers do **not** retrain models. Use the shipped checkpoint. Training / extract / upload stay on Mun Cyber lab machines (`DEVELOPER_*` in `.env`).

## Building `mun-cyber-eye-standalone.zip`

From a development checkout (not required for customers):

```powershell
python scripts/build_standalone_zip.py
```

The zip **excludes** `.venv`, `__pycache__`, `.git`, and a real `.env`. It includes `.env.example`, `install_and_run.bat`, `install_and_run.ps1`, and the bundled demo checkpoint.

CI note: run the same script on a Windows or zip-capable job if you publish a download artifact. Do not pack secrets.

## Safety

Authorized cameras only. Humans verify every alert. Face-expression assist stays off unless you enable it for lab use, and it never identifies anyone. Retention and HITL rules: [PRODUCT_OPS.md](PRODUCT_OPS.md), [ETHICS_AND_SAFETY.md](ETHICS_AND_SAFETY.md).
