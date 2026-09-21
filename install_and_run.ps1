# Mun Cyber Eye — Windows standalone launcher
# Unzip, run this script, confirm package install, then the browser opens.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host ""
Write-Host "Mun Cyber Eye — Windows standalone"
Write-Host ""
Write-Host "This installer will:"
Write-Host "  1. Create or reuse a local .venv folder"
Write-Host "  2. Install Python packages from requirements.txt into that .venv"
Write-Host "  3. Start the console at http://127.0.0.1:5055"
Write-Host "  4. Open your browser to the login / create-account page"
Write-Host ""
Write-Host "There is no default password. Create the first admin yourself (12+ characters, letter and digit)."
Write-Host "2FA is not enforced on this Capstone console."
Write-Host "Detection runs from registered cameras. Video uploads are for training only."
Write-Host ""

$confirm = Read-Host "Continue and install from requirements.txt? [Y/n]"
if ($confirm -match '^[Nn]') {
    Write-Host "Cancelled."
    exit 1
}

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ Exe = "py"; Args = @("-3") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Exe = "python"; Args = @() }
    }
    return $null
}

$pythonCmd = Find-Python
if (-not $pythonCmd) {
    Write-Host "Python 3 was not found. Install Python 3.10+ and check 'Add python.exe to PATH'."
    Read-Host "Press Enter to close"
    exit 1
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating .venv ..."
    & $pythonCmd.Exe @($pythonCmd.Args + @("-m", "venv", ".venv"))
}

Write-Host "Installing requirements.txt into .venv ..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

$envPath = Join-Path $PSScriptRoot ".env"
$examplePath = Join-Path $PSScriptRoot ".env.example"
if (-not (Test-Path -LiteralPath $envPath) -and (Test-Path -LiteralPath $examplePath)) {
    Copy-Item -LiteralPath $examplePath -Destination $envPath
    Write-Host "Copied .env.example to .env (no secrets included)."
}

$env:MUN_OPEN_BROWSER = "1"
Write-Host ""
Write-Host "Starting Mun Cyber Eye ..."
Write-Host "Open http://127.0.0.1:5055 if the browser does not appear."
Write-Host "Create the first admin account. Close this window to stop the console."
Write-Host ""
& $venvPython (Join-Path $PSScriptRoot "run.py")
