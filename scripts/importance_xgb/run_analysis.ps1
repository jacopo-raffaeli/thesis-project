python scripts/importance_xgb/run_analysis.py `
    --ticker fbtp `
    --analysis selection `
    --sample-ratio 0.01 `
    --no-use-base-target `
    --no-use-tscv `
    --n-jobs-preprocessing 4 `
    --n-jobs-xgb 8 `
    --optuna-n-trials 5 `
    --log-level DEBUG
