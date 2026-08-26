<#
Examples

# Dry-run, pull all runs (default)
.\pull.ps1

# Real pull, all runs
.\pull.ps1 -Apply

# Dry-run, pull all runs for fbts
.\pull.ps1 -Ticker fbts

# Real pull of a specific run
.\pull.ps1 -Apply -Run run_012

# Real pull of a specific run for fbts
.\pull.ps1 -Apply -Ticker fbts -Run run_012
#>

param(
    [switch]$Apply,

    [ValidateSet("fbtp", "fbts")]
    [string]$Ticker = "fbtp",

    [string]$Run = ""
)

# Path of this wrapper
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptWin = Join-Path $scriptDir "pull.sh"

# Convert Windows path to WSL path
$drive = $scriptWin.Substring(0, 1).ToLower()
$rest = $scriptWin.Substring(2).Replace('\', '/')
$script = "/mnt/$drive$rest"

# Build argument list
$arguments = @()

if ($Apply) {
    $arguments += "--apply"
}

if ($Ticker -ne "fbtp") {
    $arguments += "--ticker"
    $arguments += $Ticker
}

wsl bash $script @arguments
