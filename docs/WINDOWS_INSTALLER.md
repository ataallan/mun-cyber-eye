# Windows Setup — Mun Cyber Eye

Customers download one file, **MunCyberEyeSetup.exe**, and install Mun Cyber Eye like a normal Windows app.

The Setup program is built with [Inno Setup](https://jrsoftware.org/isinfo.php). It ships the application, a private embeddable Python 3.12 runtime, the packages from `requirements.txt` (including OpenCV), and a folder of wheels so a later repair can run offline.

The package contains `.env.example` with empty credentials. On first launch the app copies that file and generates `FLASK_SECRET_KEY`.

## Customer install

1. Download **MunCyberEyeSetup.exe**.
2. Run it. Files install for the current user under `%LocalAppData%\MunCyberEye`. Python does not need to be installed separately.
3. Leave **Launch Mun Cyber Eye** checked. The console starts and the browser opens at **http://127.0.0.1:5055**.
4. Create the first admin account in the browser. There is no default password.
5. Next time, open the Desktop shortcut **Mun Cyber Eye**. Uninstall from the Start Menu entry **Uninstall Mun Cyber Eye**, or from Windows Settings.

The Desktop shortcut uses `app/static/img/mun-cyber-eye.ico`. A Start Menu shortcut uses the same icon. Each launch opens the local console. Close the console window to stop it.

Windows may warn that the file is unrecognized until the Setup program is code-signed. Choose **More info**, then **Run anyway**.

Uninstall removes the program files that Setup installed. A `.env` or database created after install can remain in `%LocalAppData%\MunCyberEye`. Delete that folder to remove them.

## Build MunCyberEyeSetup.exe

On a 64-bit Windows machine, from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1
```

The script:

1. Copies the application into `build\windows-installer\payload` (`scripts\stage_windows_payload.py`).
2. Downloads the embeddable CPython build named in `installer\runtime.json` (currently Python 3.12.10, `python-3.12.10-embed-amd64.zip`).
3. Enables `import site` in `python*._pth`, bootstraps pip, and installs `requirements.txt` into that runtime.
4. Downloads wheels into `payload\wheels` for offline repair.
5. Imports `flask`, `cv2`, `numpy`, `PIL`, `sklearn`, `dotenv`, and `joblib`.
6. Installs Inno Setup 6.7.3 if `ISCC.exe` is not already present, then compiles `installer\MunCyberEye.iss`.

Output:

```text
dist\MunCyberEyeSetup.exe
```

The build machine needs network access for Python, pip, and Inno Setup. Customers do not. If Inno Setup 6 is already installed, the script uses that `ISCC.exe`.

Check the project without compiling:

```powershell
python scripts\build_windows_installer.py --check
```

On Linux or macOS the same command validates the scripts, icon wiring, and payload rules, then prints the Windows build command. It does not produce `MunCyberEyeSetup.exe`.

## GitHub Actions

`.github/workflows/windows-installer.yml` runs the same PowerShell script on `windows-latest` and uploads **MunCyberEyeSetup.exe** as the artifact **MunCyberEyeSetup**.

Run it from the Actions tab with **Windows installer**, or open a pull request that changes the app, installer, or requirements. Download the artifact from the completed job. That file is the customer download.

## What the launcher does

`Start Mun Cyber Eye.bat` is the Desktop target.

- It uses `python\python.exe` from the bundled runtime when that file exists.
- A source-folder install that only has `.venv` still uses `.venv\Scripts\python.exe`.
- It runs `scripts\ensure_customer_env.py` so `.env` exists and a placeholder `FLASK_SECRET_KEY` is replaced. The key is not printed.
- When the bundled packages still import, it starts `run.py` and opens the browser. It does not ask to reinstall.
- When the bundled packages fail to import, `scripts\repair_bundled_runtime.ps1` reinstalls `requirements.txt` from `wheels` with `--no-index`.

`install_and_run.bat` remains the setup path for an unzipped source folder on a PC that already has Python 3.10 or newer. See [STANDALONE.md](STANDALONE.md). Customers run `MunCyberEyeSetup.exe`.

## Layout

| Path | Role |
| --- | --- |
| `installer/MunCyberEye.iss` | Inno Setup script: per-user dir, one Desktop shortcut, Start Menu, uninstaller |
| `installer/INSTALL.txt` | Short text shown before install |
| `installer/runtime.json` | Pinned Python and Inno Setup download URLs |
| `installer/VERSION` | `major.minor.patch` written into the Setup program and into `BUILD_EPOCH` |
| `BUILD_EPOCH` | Written at the payload root on each build (`version` + UTC time). Not stored in git. A new Setup.exe rejects console cookies from the previous build. See [AUTH.md](AUTH.md). |
| `scripts/build_windows_installer.ps1` | Windows build |
| `scripts/build_windows_installer.py` | Project check, and the Windows entry that calls the PowerShell build |
| `scripts/stage_windows_payload.py` | Copies the app tree and refuses secrets |
| `scripts/ensure_customer_env.py` | Copies `.env.example` and generates `FLASK_SECRET_KEY` |
| `scripts/repair_bundled_runtime.ps1` | Offline package repair for an installed copy |
