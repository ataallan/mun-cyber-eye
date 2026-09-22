# Mun Cyber Eye launcher. Starts a healthy .venv without reinstalling.
# The Desktop shortcut points at "Start Mun Cyber Eye.bat", which calls the same flow.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Mun Cyber Eye needs setup before it can start."
    Write-Host "Double-click install_and_run.bat in this folder."
    Write-Host "That creates .venv, installs requirements, and adds the Desktop shortcut."
    Read-Host "Press Enter to close" | Out-Null
    exit 1
}

& $python -c "import flask, cv2, numpy, PIL, sklearn, dotenv, joblib"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "The local Python environment is missing core packages or is damaged."
    Write-Host "Double-click install_and_run.bat to repair .venv and refresh the Desktop shortcut."
    Read-Host "Press Enter to close" | Out-Null
    exit 1
}

$env:MUN_OPEN_BROWSER = "1"
Write-Host "Starting Mun Cyber Eye ..."
Write-Host "Open http://127.0.0.1:5055 if the browser does not appear."
Write-Host "Close this window to stop the console."
Write-Host ""
& $python (Join-Path $PSScriptRoot "run.py")
$code = $LASTEXITCODE
Write-Host ""
Write-Host "Mun Cyber Eye stopped."
Read-Host "Press Enter to close" | Out-Null
exit $code
