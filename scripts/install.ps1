<#
.SYNOPSIS
    MeshCompute installer (Windows): create a venv and install the `mesh` CLI
    + all Phase-1 packages. Idempotent (safe to re-run). Never needs admin,
    and installs nothing outside .venv.

.DESCRIPTION
    Windows PowerShell equivalent of scripts/install.sh. Works on both
    PowerShell 5.1 (Windows PowerShell) and PowerShell 7+.

.USAGE
    powershell -ExecutionPolicy Bypass -File scripts\install.ps1
    .\.venv\Scripts\mesh.exe node start
#>

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvDir = Join-Path $RepoRoot '.venv'
$RequiredMajor = 3
$RequiredMinor = 12

# --- find a python3.12+ interpreter -----------------------------------------
$Python = $null
$Candidates = @('py -3.12', 'py -3', 'python3.12', 'python3', 'python')

foreach ($candidate in $Candidates) {
    $parts = $candidate.Split(' ')
    $cmdName = $parts[0]
    $cmdArgs = @()
    if ($parts.Length -gt 1) {
        $cmdArgs = $parts[1..($parts.Length - 1)]
    }

    $found = Get-Command $cmdName -ErrorAction SilentlyContinue
    if (-not $found) {
        continue
    }

    $versionArgs = $cmdArgs + @('-c', "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
    $version = ''
    try {
        $version = & $cmdName @versionArgs 2>$null
    } catch {
        $version = ''
    }

    # Windows Store "python" alias prints nothing (or opens the store) —
    # treat empty output as not-found.
    if ([string]::IsNullOrWhiteSpace($version)) {
        continue
    }

    $versionParts = ($version | Select-Object -Last 1).ToString().Trim().Split('.')
    if ($versionParts.Length -lt 2) {
        continue
    }

    $major = 0
    $minor = 0
    $majorOk = [int]::TryParse($versionParts[0], [ref]$major)
    $minorOk = [int]::TryParse($versionParts[1], [ref]$minor)
    if (-not $majorOk -or -not $minorOk) {
        continue
    }

    if (($major -gt $RequiredMajor) -or (($major -eq $RequiredMajor) -and ($minor -ge $RequiredMinor))) {
        $Python = $candidate
        break
    }
}

if (-not $Python) {
    [Console]::Error.WriteLine("error: need python$RequiredMajor.$RequiredMinor+ on PATH (found none that qualifies).")
    [Console]::Error.WriteLine("       install Python $RequiredMajor.$RequiredMinor from python.org (or: winget install Python.Python.3.12) and re-run this script.")
    exit 1
}

$PyParts = $Python.Split(' ')
$PyCmd = $PyParts[0]
$PyArgs = @()
if ($PyParts.Length -gt 1) {
    $PyArgs = $PyParts[1..($PyParts.Length - 1)]
}

$VersionOutArgs = $PyArgs + @('--version')
$VersionOut = & $PyCmd @VersionOutArgs 2>&1
Write-Host "using $VersionOut ($Python)"

# --- create the venv (idempotent: skip if it already looks valid) ----------
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'

if (Test-Path $VenvPy) {
    Write-Host "venv already exists at $VenvDir - reusing it"
} else {
    Write-Host "creating venv at $VenvDir"
    $VenvCreateArgs = $PyArgs + @('-m', 'venv', $VenvDir)
    & $PyCmd @VenvCreateArgs
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("error: failed to create venv at $VenvDir")
        exit 1
    }
}

# --- install the package (editable) -----------------------------------------
# Prefer uv if the user already has it (faster); fall back to the venv's own
# pip otherwise. Either way, nothing is installed outside the venv.
$Uv = Get-Command uv -ErrorAction SilentlyContinue

if ($Uv) {
    Write-Host "installing with uv..."
    & uv pip install --python $VenvPy -e $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("error: uv pip install failed")
        exit 1
    }
} else {
    Write-Host "installing with pip..."
    & $VenvPy -m pip install --upgrade pip | Out-Null
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("error: pip upgrade failed")
        exit 1
    }
    & $VenvPy -m pip install -e $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("error: pip install -e failed")
        exit 1
    }
}

Write-Host ""
Write-Host "done. activate the venv and start a node:"
Write-Host ""
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    mesh node start"
Write-Host ""
Write-Host "(or run it without activating: .\.venv\Scripts\mesh.exe node start)"
Write-Host ""
Write-Host "if Activate.ps1 is blocked, run: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
Write-Host ""
Write-Host "see docs\QUICKSTART.md for contribution flags and the full-mesh flow."
