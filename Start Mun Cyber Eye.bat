@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Mun Cyber Eye

set "PY="
set "BUNDLED=0"
if exist "%~dp0python\python.exe" (
  set "PY=%~dp0python\python.exe"
  set "BUNDLED=1"
)
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" (
  set "PY=%~dp0.venv\Scripts\python.exe"
)
if not defined PY (
  echo Mun Cyber Eye needs setup before it can start.
  echo Run MunCyberEyeSetup.exe, or double-click install_and_run.bat in a source folder.
  echo.
  pause
  exit /b 1
)

if exist "%~dp0scripts\ensure_customer_env.py" (
  "%PY%" "%~dp0scripts\ensure_customer_env.py"
  if errorlevel 1 (
    echo Could not prepare local settings.
    echo.
    pause
    exit /b 1
  )
)

"%PY%" -c "import flask, cv2, numpy, PIL, sklearn, dotenv, joblib" >nul 2>&1
if not errorlevel 1 goto imports_ok
if not "%BUNDLED%"=="1" goto imports_failed

echo Repairing Mun Cyber Eye ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\repair_bundled_runtime.ps1"
"%PY%" -c "import flask, cv2, numpy, PIL, sklearn, dotenv, joblib" >nul 2>&1
if not errorlevel 1 goto imports_ok

:imports_failed
echo The local Python environment is missing core packages or is damaged.
if "%BUNDLED%"=="1" (
  echo Run MunCyberEyeSetup.exe again to repair this install.
) else (
  echo Double-click install_and_run.bat to repair .venv and refresh the Desktop shortcut.
)
echo.
pause
exit /b 1

:imports_ok
set "MUN_OPEN_BROWSER=1"
echo Starting Mun Cyber Eye ...
echo Open http://127.0.0.1:5055 if the browser does not appear.
echo Close this window to stop the console.
echo.
"%PY%" "%~dp0run.py"
set "ERR=%ERRORLEVEL%"
echo.
echo Mun Cyber Eye stopped.
pause
exit /b %ERR%
