# Build dist\MunCyberEyeSetup.exe on Windows.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1
#
# Stages the app, downloads the pinned embeddable Python, installs
# requirements.txt into that runtime, vendors wheels, and compiles
# installer\MunCyberEye.iss with Inno Setup.

$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$File,
        [string[]]$ArgumentList = @()
    )
    & $File @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (${LASTEXITCODE}): $File $($ArgumentList -join ' ')"
    }
}

function Get-BuilderPython {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ File = "py"; Prefix = @("-3") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ File = "python"; Prefix = @() }
    }
    throw "Python 3 is required on the build machine. Customers receive the bundled runtime inside Setup.exe."
}

function Invoke-BuilderPython {
    param([string[]]$PythonArgs)
    $launcher = Get-BuilderPython
    Invoke-Native -File $launcher.File -ArgumentList @($launcher.Prefix + $PythonArgs)
}

$runtimePath = Join-Path $Root "installer\runtime.json"
$runtime = Get-Content -LiteralPath $runtimePath -Raw -Encoding UTF8 | ConvertFrom-Json
$version = (Get-Content -LiteralPath (Join-Path $Root "installer\VERSION") -Raw -Encoding UTF8).Trim()
if ($version -notmatch '^\d+\.\d+\.\d+$') {
    throw "installer\VERSION must be major.minor.patch"
}

$work = Join-Path $Root "build\windows-installer"
$cache = Join-Path $work "cache"
$payload = Join-Path $work "payload"
$pythonDir = Join-Path $payload "python"
New-Item -ItemType Directory -Force -Path $cache | Out-Null
if (Test-Path -LiteralPath $payload) {
    Remove-Item -LiteralPath $payload -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $payload | Out-Null

Write-Host "Staging application files ..."
Invoke-BuilderPython -PythonArgs @(
    (Join-Path $Root "scripts\stage_windows_payload.py"),
    "--dest",
    $payload
)
$leakedEnv = @(Get-ChildItem -LiteralPath $payload -Recurse -Force -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq ".env" })
if ($leakedEnv.Count -gt 0) {
    throw "Refusing to package a real .env"
}

Write-Host "Downloading embeddable Python $($runtime.python_version) ..."
$zipName = Split-Path -Leaf ([uri]$runtime.python_embed_url).AbsolutePath
$zipPath = Join-Path $cache $zipName
if (-not (Test-Path -LiteralPath $zipPath)) {
    Invoke-Native -File "curl.exe" -ArgumentList @(
        "-fL", "--retry", "3", "--retry-delay", "2", "--ssl-no-revoke",
        "-o", $zipPath, [string]$runtime.python_embed_url
    )
}
if (Test-Path -LiteralPath $pythonDir) {
    Remove-Item -LiteralPath $pythonDir -Recurse -Force
}
Expand-Archive -LiteralPath $zipPath -DestinationPath $pythonDir -Force
$pythonExe = Join-Path $pythonDir "python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $nested = @(Get-ChildItem -LiteralPath $pythonDir -Recurse -Filter "python.exe" -File | Select-Object -First 1)
    if ($nested.Count -eq 1) {
        $pythonExe = $nested[0].FullName
        $pythonDir = $nested[0].DirectoryName
    }
}
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Embeddable archive did not contain python.exe"
}

$stdlibZip = @(Get-ChildItem -LiteralPath $pythonDir -Filter "python*.zip" -File)
$pthFile = @(Get-ChildItem -LiteralPath $pythonDir -Filter "python*._pth" -File)
if ($stdlibZip.Count -lt 1 -or $pthFile.Count -lt 1) {
    throw "Embeddable Python is missing python*.zip or python*._pth"
}
Write-Host "Enabling site-packages in $($pthFile[0].Name) ..."
Invoke-BuilderPython -PythonArgs @(
    (Join-Path $Root "scripts\build_windows_installer.py"),
    "--write-pth",
    $stdlibZip[0].Name,
    $pthFile[0].FullName
)
New-Item -ItemType Directory -Force -Path (Join-Path $pythonDir "Lib\site-packages") | Out-Null

$getPip = Join-Path $cache "get-pip.py"
if (-not (Test-Path -LiteralPath $getPip)) {
    Write-Host "Downloading get-pip.py ..."
    Invoke-Native -File "curl.exe" -ArgumentList @(
        "-fL", "--retry", "3", "--retry-delay", "2", "--ssl-no-revoke",
        "-o", $getPip, [string]$runtime.get_pip_url
    )
}
Write-Host "Installing pip into the bundled runtime ..."
Invoke-Native -File $pythonExe -ArgumentList @($getPip, "--no-warn-script-location")

$requirements = Join-Path $payload "requirements.txt"
Write-Host "Installing requirements.txt into the bundled runtime ..."
Invoke-Native -File $pythonExe -ArgumentList @(
    "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"
)
Invoke-Native -File $pythonExe -ArgumentList @(
    "-m", "pip", "install", "--disable-pip-version-check", "-r", $requirements
)

$wheels = Join-Path $payload "wheels"
New-Item -ItemType Directory -Force -Path $wheels | Out-Null
Write-Host "Downloading offline wheels ..."
Invoke-Native -File $pythonExe -ArgumentList @(
    "-m", "pip", "download", "--disable-pip-version-check", "-r", $requirements, "-d", $wheels
)

Write-Host "Checking core imports ..."
Push-Location -LiteralPath $env:TEMP
try {
    Invoke-Native -File $pythonExe -ArgumentList @(
        "-c", "import flask, cv2, numpy, PIL, sklearn, dotenv, joblib"
    )
} finally {
    Pop-Location
}

Get-ChildItem -LiteralPath $payload -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath $payload -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

function Find-Iscc {
    $roots = @(${env:ProgramFiles(x86)}, $env:ProgramFiles) | Where-Object { $_ }
    foreach ($base in $roots) {
        $candidate = Join-Path $base "Inno Setup 6\ISCC.exe"
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $null
}

$iscc = Find-Iscc
if (-not $iscc) {
    Write-Host "Installing Inno Setup $($runtime.inno_version) ..."
    $innoInstaller = Join-Path $cache ("innosetup-" + $runtime.inno_version + ".exe")
    if (-not (Test-Path -LiteralPath $innoInstaller)) {
        Invoke-Native -File "curl.exe" -ArgumentList @(
            "-fL", "--retry", "3", "--retry-delay", "2", "--ssl-no-revoke",
            "-o", $innoInstaller, [string]$runtime.inno_url
        )
    }
    $install = Start-Process -FilePath $innoInstaller -ArgumentList @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/SP-"
    ) -Wait -PassThru
    if ($install.ExitCode -ne 0 -and $install.ExitCode -ne 3010) {
        throw "Inno Setup installer exited with $($install.ExitCode)"
    }
    $iscc = Find-Iscc
}
if (-not $iscc) {
    throw "ISCC.exe was not found after installing Inno Setup"
}

$iss = Join-Path $Root "installer\MunCyberEye.iss"
Write-Host "Compiling $iss ..."
Invoke-Native -File $iscc -ArgumentList @(
    $iss,
    "/DPayloadDir=../build/windows-installer/payload",
    "/DAppVersion=$version"
)

$out = Join-Path $Root ("dist\" + $runtime.output_exe)
if (-not (Test-Path -LiteralPath $out)) {
    throw "Inno Setup did not write $out"
}
$item = Get-Item -LiteralPath $out
Write-Host "Wrote $($item.FullName) ($([math]::Round($item.Length / 1MB, 1)) MB)"
