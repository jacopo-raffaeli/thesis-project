#!/usr/bin/env bash

python scripts/importance_xgb/run_analysis.py \
    --ticker fbtp \
    --analysis selection \
    --sample-ratio 0.05 \
    --no-use-base-target \
    --no-use-tscv \
    --n-jobs-preprocessing 12 \
    --n-jobs-xgb -1 \
    --optuna-n-trials 150 \
    --log-level DEBUG
