#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOCAL="/mnt/c/Users/Jacopo/Desktop/thesis/thesis-project/"
REMOTE="intesa@10.75.4.18:~/jacopo/thesis-project/"
FILTER_FILE="$SCRIPT_DIR/rsync_filter.txt"

# To dry run use -avzn in place ov -avz
rsync \
    -avz \
    --info=progress2 \
    --filter="merge $FILTER_FILE" \
    "$REMOTE" \
    "$LOCAL"
