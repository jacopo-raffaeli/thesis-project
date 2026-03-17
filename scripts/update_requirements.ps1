<#
.SYNOPSIS
    Regenerate requirements.txt from imports and pin versions from local .venv.

.DESCRIPTION
    Uses pipreqs to scan .py and .ipynb files and emit package names only.
    Resolves each package version from the local virtual environment.
    Writes a pinned requirements.txt.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Python interpreter not found at $venvPython"
}

Write-Host "Scanning imports with pipreqs..." -ForegroundColor Cyan
& $venvPython -m pipreqs.pipreqs --force --ignore .venv --scan-notebooks --mode no-pin .

if (-not (Test-Path "requirements.txt")) {
    throw "requirements.txt was not created by pipreqs"
}

$rawPackages = Get-Content "requirements.txt" |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.StartsWith("#") }

if ($rawPackages.Count -eq 0) {
    Write-Warning "No third-party imports found. requirements.txt is empty."
    exit 0
}

$packages = $rawPackages | Sort-Object -Unique
$pinned = New-Object System.Collections.Generic.List[string]
$unresolved = New-Object System.Collections.Generic.List[string]

Write-Host "Resolving installed versions from local .venv..." -ForegroundColor Cyan
foreach ($pkg in $packages) {
    $showOutput = & $venvPython -m pip show $pkg 2>$null
    if (-not $showOutput) {
        $unresolved.Add($pkg)
        continue
    }

    $versionLine = $showOutput | Select-String '^Version:' | Select-Object -First 1
    if (-not $versionLine) {
        $unresolved.Add($pkg)
        continue
    }

    $version = ($versionLine.Line -split ':', 2)[1].Trim()
    if (-not $version) {
        $unresolved.Add($pkg)
        continue
    }

    $pinned.Add("$pkg==$version")
}

Write-Host "Writing requirements.txt..." -ForegroundColor Cyan
$pinned | Sort-Object | Set-Content "requirements.txt"

if ($unresolved.Count -gt 0) {
    Write-Warning "Could not resolve local versions for: $($unresolved -join ', ')"
    Write-Warning "They were omitted from requirements.txt; install them in .venv and re-run."
}

Write-Host "Done. requirements.txt updated with $($pinned.Count) pinned package(s)." -ForegroundColor Green
