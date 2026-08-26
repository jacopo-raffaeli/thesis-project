#!/usr/bin/env bash

set -euo pipefail

LOCAL="/mnt/c/Users/Jacopo/Desktop/thesis/thesis-project"
REMOTE_ROOT="intesa@10.75.4.18:~/jacopo/thesis-project"

# Defaults
DRY_RUN=true
TICKER="fbtp"

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply)
            DRY_RUN=false
            shift
            ;;
        --ticker)
            TICKER="$2"

            case "$TICKER" in
                fbtp|fbts) ;;
                *)
                    echo "Error: invalid ticker '$TICKER'. Valid values are: fbtp, fbts."
                    exit 1
                    ;;
            esac

            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# rsync flags
RSYNC_FLAGS="-avz"
$DRY_RUN && RSYNC_FLAGS="${RSYNC_FLAGS}n"

# Determine analysis folder from the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE_RUNS="$REMOTE_ROOT/results/experiments/$TICKER/profitability"
LOCAL_RUNS="$LOCAL/results/experiments/$TICKER/profitability"

rsync \
    $RSYNC_FLAGS \
    --info=progress2 \
    "$REMOTE_RUNS/" \
    "$LOCAL_RUNS/"
