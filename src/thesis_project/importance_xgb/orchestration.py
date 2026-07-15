import logging
from dataclasses import asdict
from functools import partial

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from thesis_project import config as global_config
from thesis_project.importance_xgb import (
    config,
    data_paths,
    data_pipeline,
    importance,
    tuning,
    utils,
)

logger = logging.getLogger(__name__)


def run_analysis(config_obj: config.AnalysisConfig):
    # Get experiment number and output path
    n_run = utils.generate_run_number(config_obj)
    output_path = utils.generate_run_path(config_obj)

    logger.info(f"""

    XGBoost importance analysis:

    Ticker:                 {config_obj.ticker}
    Time window:            {config_obj.min_time} - {config_obj.max_time}
    Sample ratio:           {config_obj.sampling_params["sample_ratio"]}
    Use base target:        {config_obj.use_base_target}
    Use tscv:               {config_obj.use_tscv}
    N jobs preprocessing:   {config_obj.n_jobs_preprocessing}
    N jobs xgboost:         {config_obj.n_jobs_xgb}
    N seeds:                {config_obj.optuna_n_seeds}
    N jobs seed:            {config_obj.n_jobs_seed}
    Quantiles:              {config_obj.n_quantile}
    Optuna trials:          {config_obj.optuna_n_trials}

    Run:                    {n_run:03d}
    Output path:            {output_path.relative_to(global_config.ROOT)}

    """)

    # Save experiment config
    with open(output_path / "config.json", "w") as f:
        yaml.safe_dump(asdict(config_obj), f, sort_keys=False)
    logger.info("Experiment config saved to '%s'", "config.json")

    # Get feature and target data paths
    paths = data_paths.get_data_paths(config_obj.ticker)

    # Save experiment data paths
    with open(output_path / "data_paths.yaml", "w") as f:
        k, v = paths.target
        paths_dict = {
            "target": {k: str(v)},
            "features": {k: str(v) for k, v in paths.features.items()},
        }
        yaml.safe_dump(paths_dict, f, sort_keys=False)
    logger.info("Experiment data paths saved to '%s'", "data_paths.yaml")

    # Generate seeds
    rng = np.random.default_rng(np.random.randint(0, 9999))
    seeds = rng.integers(0, 9999, size=config_obj.optuna_n_seeds)

    # Load target
    (target_name, target_path) = paths.target
    target_df = data_pipeline.load_data(target_name, target_path)
    logger.info(
        "Base target '%s' loaded from '%s'",
        target_name,
        target_path.relative_to(global_config.ROOT),
    )

    # Preprocess target
    logger.info("Preprocessing target")
    y, idx_sampled, dates_sampled = data_pipeline.preprocess_target(
        config_obj,
        target_df,
    )
    logger.info("Target preprocessed")

    # Preprocess market features
    logger.info("Preprocessing features")
    X = data_pipeline.preprocess_features(config_obj, idx_sampled, dates_sampled)
    logger.info("Features preprocessed")

    # Match features and target
    X, y = data_pipeline.match_X_y(config_obj, X, y)
    logger.info("Target and features datasets aligned")

    # Split data
    X_Y_split = data_pipeline.split_data(config_obj, X, y)
    logger.info("Dataset splitted for training")

    # Extract the features sets
    X_train = X_Y_split["train"]["features"]
    X_val = X_Y_split["validation"]["features"]
    X_test = X_Y_split["test"]["features"]
    X_train_val = pd.concat([X_train, X_val])

    # Extract the target sets
    y_train = X_Y_split["train"]["target"]
    y_val = X_Y_split["validation"]["target"]
    y_test = X_Y_split["test"]["target"]
    y_train_val = pd.concat([y_train, y_val])

    # Compute bins for classification
    if config_obj.use_tscv:
        # If tscv is used Extract the smallest training set to compute future agnostic quantile
        splitter = TimeSeriesSplit(n_splits=config_obj.n_splits_tscv)
        splits = list(splitter.split(X_train_val))
        first_train_idxs, _ = splits[0]
        bins = data_pipeline.compute_bins(
            y_train_val.iloc[first_train_idxs, 0], config_obj.n_quantile
        )
    else:
        # If only one validation set is used compute the classification on the whole training set
        splits = [(np.arange(len(X_train)), np.arange(len(X_train), len(X_train_val)))]
        bins = data_pipeline.compute_bins(y_train.iloc[:, 0], config_obj.n_quantile)

    # Classify target
    y_cls_train = data_pipeline.classify_target(y_train, bins)
    y_cls_val = data_pipeline.classify_target(y_val, bins)
    y_cls_train_val = pd.concat([y_cls_train, y_cls_val])
    y_cls_test = data_pipeline.classify_target(y_test, bins)
    logger.info("Target classified")

    # Generate optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective_par = partial(
        tuning.objective,
        config_obj=config_obj,
        seeds=seeds,
        X=X_train_val,
        y=y_cls_train_val,
        splits=splits,
    )
    study = optuna.create_study(
        study_name=f"run_{n_run:03d}",
        storage=f"sqlite:///{output_path / 'optuna.db'}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=config_obj.seed, n_startup_trials=config_obj.optuna_sampler_n_startup_trials
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config_obj.optuna_pruner_n_startup_trials,
            n_warmup_steps=config_obj.optuna_pruner_n_warmup_steps,
        ),
    )
    logger.info("Optuna study generated")

    # Run optimization
    logger.info("Optuna study started")
    callback = partial(tuning.optuna_callback, config_obj=config_obj)
    study.optimize(
        objective_par,
        n_trials=config_obj.optuna_n_trials,
        callbacks=[callback],
        show_progress_bar=True,
    )
    logger.info("Optuna study ended")
    logger.info("Optuna study is available at '%s'", "optuna.db")

    fixed_params = config.get_xgb_fixed_params(config_obj)
    best_params = {
        **fixed_params,
        **study.best_params,
        "n_estimators": int(np.mean(study.best_trial.user_attrs["best_iterations"])) + 1,
    }

    # Retrain best model
    importance_xgb = {}
    importance_shap = {}
    logger.info("Retraining best configuration on Training + Validation data")
    for seed in seeds:
        best_model = XGBClassifier(random_state=seed, **best_params)
        best_model.fit(X_train_val, y_cls_train_val.iloc[:, 0])
        # logger.info("Best configuration retrained")

        # Save best model
        best_model.save_model(output_path / f"model_seed_{seed}.ubj")
        logger.info("Best model saved to '%s'", f"model_seed_{seed}.ubj")

        # Perform and save xgb importance analysis
        importance_xgb[seed] = importance.importance_xgb(best_model)
        importance_xgb[seed].to_csv(output_path / f"importance_xgb_seed_{seed}.csv", index=False)
        logger.info("XGBoost importance metrics saved to '%s'", f"importance_xgb.csv_seed_{seed}")

        # Perfom and save shap importance analysis
        importance_shap[seed] = importance.importance_shap(config_obj, best_model, X_test)
        joblib.dump(
            importance_shap[seed], output_path / f"importance_shap_seed_{seed}.pkl", compress=True
        )
        logger.info("Shap importance metrics saved to '%s'", f"importance_shap_{seed}.csv")

        # Save predicted probabilities
        pd.DataFrame(
            best_model.predict_proba(X_test),
            columns=[f"{y_test.columns[0]}_class_{i}" for i in range(config_obj.n_quantile)],
            index=y_test.index,
        ).to_parquet(output_path / f"y_cls_proba_test_seed_{seed}.parquet")
        logger.info(
            "Target test set predicted probabilities saved to '%s'",
            f"y_cls_proba_test_seed_{seed}.parquet",
        )

    # Save target
    y_test.to_parquet(output_path / "y_test.parquet")
    logger.info("Target test set saved to '%s'", "y_test.parquet")

    # Save classified target
    pd.DataFrame(y_cls_test, columns=[y_test.columns[0]], index=y_test.index).to_parquet(
        output_path / "y_cls_test.parquet"
    )
    logger.info("Target test set classified saved to '%s'", "y_cls_test.parquet")

    # Save metadata
    metadata = {
        "n_features": len(X.columns),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "n_quantile": config_obj.n_quantile,
        "quantile_edges": bins.tolist(),
        "seeds": seeds.tolist(),
    }
    with open(output_path / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    logger.info("Metadata saved to '%s'", "metadata.yaml")

    return 0
