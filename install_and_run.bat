@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo Mun Cyber Eye — Windows standalone
echo.
echo This installer will:
echo   1. Create or reuse a local .venv folder
echo   2. Install Python packages from requirements.txt into that .venv
echo   3. Start the console at http://127.0.0.1:5055
echo   4. Open your browser to the login / create-account page
echo.
echo There is no default password. Create the first admin yourself.
echo Detection runs from registered cameras. Video uploads are for training only.
echo.
set /p CONFIRM=Continue and install from requirements.txt? [Y/n] 
if /I "%CONFIRM%"=="n" (
  echo Cancelled.
  exit /b 1
)

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3 was not found. Install Python 3.10+ and check "Add python.exe to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating .venv ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Failed to create .venv
    pause
    exit /b 1
  )
)

echo Installing requirements.txt into .venv ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo pip upgrade failed
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package install failed
  pause
  exit /b 1
)

if not exist ".env" (
  if exist ".env.example" (
    copy /Y ".env.example" ".env" >nul
    echo Copied .env.example to .env (no secrets included).
  )
)

set "MUN_OPEN_BROWSER=1"
echo.
echo Starting Mun Cyber Eye ...
echo Open http://127.0.0.1:5055 if the browser does not appear.
echo Create the first admin account. Close this window to stop the console.
echo.
".venv\Scripts\python.exe" run.py
set "ERR=%ERRORLEVEL%"
echo.
pause
exit /b %ERR%
