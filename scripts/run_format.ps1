<#
.SYNOPSIS
    Install and run all pre-commit hooks on every file in the repo.

.DESCRIPTION
    Run from the repo root with the project venv already activated.
    On first use it installs pre-commit (if missing) and registers the git hooks.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Get-Command pre-commit -ErrorAction SilentlyContinue)) {
    Write-Host "pre-commit not found — installing..." -ForegroundColor Yellow
    pip install pre-commit
}

Write-Host "Installing git hooks..." -ForegroundColor Cyan
pre-commit install

Write-Host "Running all hooks on all files..." -ForegroundColor Cyan
pre-commit run --all-files
