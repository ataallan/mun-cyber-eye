# Mun Cyber Eye launcher. Starts a healthy runtime without reinstalling.
# Prefer the bundled embeddable Python from Setup. Otherwise use .venv.

$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$bundled = Join-Path $PSScriptRoot "python\python.exe"
$venv = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$python = $null
$bundledInstall = $false
if (Test-Path -LiteralPath $bundled) {
    $python = $bundled
    $bundledInstall = $true
} elseif (Test-Path -LiteralPath $venv) {
    $python = $venv
}

if (-not $python) {
    Write-Host "Mun Cyber Eye needs setup before it can start."
    Write-Host "Run MunCyberEyeSetup.exe, or double-click install_and_run.bat in a source folder."
    Read-Host "Press Enter to close" | Out-Null
    exit 1
}

$ensure = Join-Path $PSScriptRoot "scripts\ensure_customer_env.py"
if (Test-Path -LiteralPath $ensure) {
    & $python $ensure
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Could not prepare local settings."
        Read-Host "Press Enter to close" | Out-Null
        exit 1
    }
}

& $python -c "import flask, cv2, numpy, PIL, sklearn, dotenv, joblib"
$importsOk = $LASTEXITCODE -eq 0
if (-not $importsOk -and $bundledInstall) {
    Write-Host "Repairing Mun Cyber Eye ..."
    $repair = Join-Path $PSScriptRoot "scripts\repair_bundled_runtime.ps1"
    if (Test-Path -LiteralPath $repair) {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $repair
    }
    & $python -c "import flask, cv2, numpy, PIL, sklearn, dotenv, joblib"
    $importsOk = $LASTEXITCODE -eq 0
}
if (-not $importsOk) {
    Write-Host ""
    Write-Host "The local Python environment is missing core packages or is damaged."
    if ($bundledInstall) {
        Write-Host "Run MunCyberEyeSetup.exe again to repair this install."
    } else {
        Write-Host "Double-click install_and_run.bat to repair .venv and refresh the Desktop shortcut."
    }
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
