@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Mun Cyber Eye

if not exist ".venv\Scripts\python.exe" (
  echo Mun Cyber Eye needs setup before it can start.
  echo Double-click install_and_run.bat in this folder.
  echo That creates .venv, installs requirements, and adds the Desktop shortcut.
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -c "import flask, cv2, numpy, PIL, sklearn, dotenv, joblib" >nul 2>&1
if errorlevel 1 (
  echo The local Python environment is missing core packages or is damaged.
  echo Double-click install_and_run.bat to repair .venv and refresh the Desktop shortcut.
  echo.
  pause
  exit /b 1
)

set "MUN_OPEN_BROWSER=1"
echo Starting Mun Cyber Eye ...
echo Open http://127.0.0.1:5055 if the browser does not appear.
echo Close this window to stop the console.
echo.
".venv\Scripts\python.exe" "%~dp0run.py"
set "ERR=%ERRORLEVEL%"
echo.
echo Mun Cyber Eye stopped.
pause
exit /b %ERR%
