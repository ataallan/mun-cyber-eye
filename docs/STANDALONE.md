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

1. Unzip the folder. `C:\MunCyberEye` is a reliable location. A Desktop folder synced by OneDrive can lock files while Python builds `.venv`.
2. Double-click **`install_and_run.bat`** (or right-click **`install_and_run.ps1`** → Run with PowerShell). The `.bat` starts the PowerShell installer with execution policy bypass.
3. Confirm the prompt. The installer creates `.venv`, installs `requirements.txt`, and writes **`install.log`**.
4. After a successful install it creates two shortcuts named **Mun Cyber Eye** (Desktop, and one inside the product folder), then starts the console at **http://127.0.0.1:5055** and opens the browser to login / create-account.
5. Create the first site admin account (cameras, alerts, recipients). It is auto-approved. Sign in with the password you chose, then enter the email sign-in code (or the on-page demo code if `AUTH_SHOW_LOGIN_CODE=1` and Resend is not configured). Later Create account users are operators and stay pending until you approve them on **Accounts**. Neither role can train models.

Later, double-click the Desktop shortcut **Mun Cyber Eye**. That runs `Start Mun Cyber Eye.bat` and opens the console. A second double-click of `install_and_run.bat` sees a healthy `.venv` and offers **Start only** (press Enter) or **U** to upgrade packages.

Windows `.bat` files keep the default script icon. The clickable branded file is the **Mun Cyber Eye** shortcut (`.lnk`) on the Desktop or in the product folder. `start_eye.ps1` is the same launcher for a PowerShell window.

If `.env` is missing, the launcher copies `.env.example`. That file has **empty** `RESEND_API_KEY` / `RESEND_FROM` placeholders. Camera owners get alerts via each account’s `security_email`, then login email — you do not put secrets in the zip.

Do not commit a real `.env`. Do not ship `.venv`.

### Icon and shortcuts

`app/static/img/mun-cyber-eye.ico` is a multi-size icon (16, 24, 32, 48, 64, 128, and 256). `scripts/build_icon.py` builds it from `app/static/img/mun-cyber-eye-logo.png`: a square crop of the eye, on the console background color. The customer zip already contains the `.ico`.

When `.venv` is healthy, `install_and_run.ps1` creates the shortcuts with `WScript.Shell`:

| Field | Value |
| --- | --- |
| Name | Mun Cyber Eye |
| Target | `Start Mun Cyber Eye.bat` in the install folder |
| WorkingDirectory | the install folder |
| IconLocation | `app\static\img\mun-cyber-eye.ico,0` |

One shortcut is saved on the user Desktop (`[Environment]::GetFolderPath("Desktop")`, which follows a redirected Desktop). The other is saved in the product folder so the unzipped package shows **Mun Cyber Eye** with the eye icon. Shortcuts store absolute paths, so they are created on the customer PC. The zip ships the `.ico` and the launcher scripts.

### If install fails

- **Python missing, or older than 3.10.** Install [Python 3.10+](https://www.python.org/downloads/) and check **Add python.exe to PATH**, then run `install_and_run.bat` again.
- **pip or package errors.** Read `install.log` in the install folder. The window also prints that path.
- **Damaged `.venv`.** Interrupted pip leftovers such as `~orch` (torch renamed mid-install) or a `.dist-info` folder missing `METADATA` make the installer delete `.venv` and install `requirements.txt` again. A healthy check imports `flask`, `cv2`, `numpy`, `PIL`, `sklearn`, `dotenv`, and `joblib`.
- **Optional YOLO.** Ultralytics and torch are left out of `requirements.txt`. When those leftovers are the only damage and the core packages still import, the installer deletes the leftovers and continues.
- **OneDrive or Desktop file locks.** If setup fails in a Desktop or OneDrive folder, copy the unzipped folder to `C:\MunCyberEye` and run `install_and_run.bat` there.

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

The zip **includes** `.env.example`, `install_and_run.bat`, `install_and_run.ps1`, `Start Mun Cyber Eye.bat`, `start_eye.ps1`, `app/static/img/mun-cyber-eye.ico`, and the bundled demo checkpoint. It **excludes** `.venv`, `__pycache__`, `.git`, a real `.env`, `install.log`, and machine-specific `.lnk` shortcuts.

CI note: run the same script on a Windows or zip-capable job if you publish a download artifact. Do not pack secrets.

## Safety

Authorized cameras only. Humans verify every alert. Face-expression assist stays off unless you enable it for lab use, and it never identifies anyone. Retention and HITL rules: [PRODUCT_OPS.md](PRODUCT_OPS.md), [ETHICS_AND_SAFETY.md](ETHICS_AND_SAFETY.md).
