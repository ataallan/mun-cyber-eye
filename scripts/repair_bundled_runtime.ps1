# Reinstall requirements into the bundled embeddable Python.
# Prefer the offline wheels shipped with Setup. Leave .env and databases alone.

$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

$Log = Join-Path $Root "install.log"
$Python = Join-Path $Root "python\python.exe"
$Requirements = Join-Path $Root "requirements.txt"
$Wheels = Join-Path $Root "wheels"

function Write-RepairLog([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Bundled Python is missing. Run MunCyberEyeSetup.exe again."
    exit 1
}
if (-not (Test-Path -LiteralPath $Requirements)) {
    Write-Host "requirements.txt is missing. Run MunCyberEyeSetup.exe again."
    exit 1
}

$wheelFiles = @()
if (Test-Path -LiteralPath $Wheels) {
    $wheelFiles = @(Get-ChildItem -LiteralPath $Wheels -Filter *.whl -File -ErrorAction SilentlyContinue)
}

$pipArgs = @("install", "--disable-pip-version-check", "-r", $Requirements)
if ($wheelFiles.Count -gt 0) {
    $pipArgs = @(
        "install",
        "--disable-pip-version-check",
        "--no-index",
        "--find-links",
        $Wheels,
        "-r",
        $Requirements
    )
    Write-RepairLog "repair from offline wheels"
} else {
    Write-RepairLog "repair from requirements.txt (no offline wheels found)"
}

Write-Host "Restoring Mun Cyber Eye packages ..."
$code = 1
$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Python -m pip @pipArgs 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
        Write-Host $line
    }
    $code = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previous
}

if ($code -ne 0) {
    Write-RepairLog "repair failed with exit $code"
    Write-Host "Repair failed. Run MunCyberEyeSetup.exe again."
    exit 1
}

Write-RepairLog "repair finished"
exit 0
