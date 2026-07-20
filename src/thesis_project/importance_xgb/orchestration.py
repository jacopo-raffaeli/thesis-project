import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

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
    if config_obj.analysis == "importance":
        run_analysis_importance(config_obj)
    elif config_obj.analysis == "selection":
        run_analysis_selection(config_obj)

    return 0


def run_analysis_importance(config_obj: config.AnalysisConfig):
    # Get experiment number and output path
    n_run = utils.generate_run_number(config_obj)
    output_path = utils.generate_run_path(config_obj)

    # Get feature and target data paths
    paths = data_paths.get_data_paths(config_obj.ticker)

    logger.info(f"""

    XGBoost feature importance analysis:

    Ticker:                 {config_obj.ticker}
    Analysis:               {config_obj.analysis}
    Time window:            {config_obj.min_time} - {config_obj.max_time}
    Sample ratio:           {config_obj.sampling_params["sample_ratio"]}
    N seeds:                {config_obj.optuna_n_seeds}
    Use base target:        {config_obj.use_base_target}
    Use tscv:               {config_obj.use_tscv}
    N jobs preprocessing:   {config_obj.n_jobs_preprocessing}
    N jobs xgboost:         {config_obj.n_jobs_xgb}
    N jobs seed:            {config_obj.n_jobs_seed}
    Quantiles:              {config_obj.n_quantile}
    Optuna trials:          {config_obj.optuna_n_trials}

    Run:                    {n_run:03d}
    Output path:            {output_path.relative_to(global_config.ROOT)}

    """)

    # Save experiment config
    filename = "config.yaml"
    _save_config(config_obj, output_path, filename)
    logger.info("Experiment config saved to '%s'", filename)

    # Save experiment data paths
    filename = "data_paths.yaml"
    _save_paths(paths, output_path, filename)
    logger.info("Experiment data paths saved to '%s'", filename)

    # Generate seeds
    rng = np.random.default_rng(np.random.randint(0, 9999))
    seeds = rng.integers(0, 9999, size=config_obj.optuna_n_seeds)

    # Prepare dataset
    data_splits = _prepare_data(config_obj, paths)

    # Extract the features sets
    X_train = data_splits["train"]["features"]
    X_val = data_splits["validation"]["features"]
    X_test = data_splits["test"]["features"]
    X_train_val = pd.concat([X_train, X_val])

    # Extract the target sets
    y_train = data_splits["train"]["target"]
    y_val = data_splits["validation"]["target"]
    y_test = data_splits["test"]["target"]
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

    # Sample subset for shap analysis (for computational reasons)
    rng = np.random.default_rng(config_obj.seed)
    shap_n_samples = min(config_obj.shap_n_samples, len(X_test))
    idx = rng.choice(
        X_test.index,
        size=shap_n_samples,
        replace=False,
    )
    X_test_shap = X_test.loc[idx]

    # Retrain best model
    logger.info("Retraining best configuration on Training + Validation data")
    Parallel(n_jobs=config_obj.n_jobs_seed, backend="loky")(
        delayed(_train_best_model)(
            config_obj=config_obj,
            X_train=X_train_val,
            y_train=y_cls_train_val,
            X_test=X_test,
            y_test=y_test,
            X_test_shap=X_test_shap,
            params=best_params,
            seed=seed,
            path=output_path,
        )
        for seed in seeds
    )

    # Save target
    filename = "y_test.parquet"
    y_test.to_parquet(output_path / filename)
    logger.info("Target test set saved to '%s'", filename)

    # Save classified target
    filename = "y_cls_test.parquet"
    pd.DataFrame(y_cls_test, columns=[y_test.columns[0]], index=y_test.index).to_parquet(
        output_path / filename
    )
    logger.info("Target test set classified saved to '%s'", filename)

    # Save metadata
    filename = "metadata.yaml"
    metadata = {
        "n_features": len(X_train.columns),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "test_shap_size": len(X_test_shap),
        "n_quantile": config_obj.n_quantile,
        "quantile_edges": bins.tolist(),
        "seeds": seeds.tolist(),
    }
    with open(output_path / filename, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    logger.info("Metadata saved to '%s'", filename)

    return 0


def run_analysis_selection(config_obj: config.AnalysisConfig):
    # Get experiment number and output path
    n_run = utils.generate_run_number(config_obj)
    output_path = utils.generate_run_path(config_obj)

    # Get feature and target data paths
    paths = data_paths.get_data_paths(config_obj.ticker)

    logger.info(f"""

    XGBoost feature selection analysis:

    Ticker:                 {config_obj.ticker}
    Analysis:               {config_obj.analysis}
    Time window:            {config_obj.min_time} - {config_obj.max_time}
    Sample ratio:           {config_obj.sampling_params["sample_ratio"]}
    N seeds:                {config_obj.optuna_n_seeds}
    Use base target:        {config_obj.use_base_target}
    Use tscv:               {config_obj.use_tscv}
    N jobs preprocessing:   {config_obj.n_jobs_preprocessing}
    N jobs xgboost:         {config_obj.n_jobs_xgb}
    N jobs seed:            {config_obj.n_jobs_seed}
    Quantiles:              {config_obj.n_quantile}
    Optuna trials:          {config_obj.optuna_n_trials}

    Run:                    {n_run:03d}
    Output path:            {output_path.relative_to(global_config.ROOT)}

    """)

    # Save experiment config
    filename = "config.yaml"
    _save_config(config_obj, output_path, filename)
    logger.info("Experiment config saved to '%s'", filename)

    # Save experiment data paths
    filename = "data_paths.yaml"
    _save_paths(paths, output_path, filename)
    logger.info("Experiment data paths saved to '%s'", filename)

    # Generate seeds
    rng = np.random.default_rng(np.random.randint(0, 9999))
    seeds = rng.integers(0, 9999, size=config_obj.optuna_n_seeds)

    # Prepare dataset
    data_splits = _prepare_data(config_obj, paths)

    # Extract the features sets
    X_train = data_splits["train"]["features"]
    X_val = data_splits["validation"]["features"]
    X_test = data_splits["test"]["features"]
    X_train_val = pd.concat([X_train, X_val])

    # Extract the target sets
    y_train = data_splits["train"]["target"]
    y_val = data_splits["validation"]["target"]
    y_test = data_splits["test"]["target"]
    y_train_val = pd.concat([y_train, y_val])

    # Sample subset for shap analysis (for computational reasons)
    rng = np.random.default_rng(config_obj.seed)
    shap_n_samples = min(config_obj.shap_n_samples, len(X_test))
    idx = rng.choice(
        X_test.index,
        size=shap_n_samples,
        replace=False,
    )
    X_test_shap = X_test.loc[idx]

    fixed_params = config.get_xgb_fixed_params(config_obj)

    Parallel(n_jobs=config_obj.n_jobs_seed, backend="loky")(
        delayed(_feature_selection_loop)(
            config_obj=config_obj,
            X_train=X_train_val,
            y_train=y_train_val,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_test_shap=X_test_shap,
            params=fixed_params,
            seed=seed,
            path=output_path,
        )
        for seed in seeds
    )

    # Save target
    filename = "y_test.parquet"
    y_test.to_parquet(output_path / filename)
    logger.info("Target test set saved to '%s'", filename)

    # Save metadata
    filename = "metadata.yaml"
    metadata = {
        "n_features": len(X_train.columns),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "test_shap_size": len(X_test_shap),
        "seeds": seeds.tolist(),
    }
    with open(output_path / filename, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    logger.info("Metadata saved to '%s'", filename)

    return 0


def _prepare_data(
    config_obj: config.AnalysisConfig, paths: data_paths.DataPathsConfig
) -> dict[str, dict[str, pd.DataFrame]]:
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
    y = data_pipeline.preprocess_target(
        config_obj,
        target_df,
    )
    logger.info("Target preprocessed")

    # Sample timestamps
    assert isinstance(y.index, pd.DatetimeIndex)
    idx_sampled, dates_sampled = data_pipeline.preprocess_timestamps(config_obj, y.index)

    # Retain only sampled target
    y = y.loc[idx_sampled]

    # Preprocess market features
    logger.info("Preprocessing features")
    X = data_pipeline.preprocess_features(config_obj, idx_sampled, dates_sampled)
    logger.info("Features preprocessed")

    # Match features and target
    X, y = data_pipeline.match_X_y(config_obj, X, y)
    logger.info("Target and features datasets aligned")

    # Split data
    X_y_split = data_pipeline.split_data(config_obj, X, y)
    logger.info("Dataset splitted for training")

    return X_y_split


def _train_best_model(
    config_obj: config.AnalysisConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    X_test_shap: pd.DataFrame,
    params,
    seed,
    path,
):
    # Retrain best model
    best_model = XGBClassifier(random_state=seed, **params)
    best_model.fit(X_train, y_train.iloc[:, 0])

    # Save best model
    filename = f"model_seed_{seed}.ubj"
    best_model.save_model(path / filename)
    logger.info("Best model saved to '%s'", filename)

    # Perform and save xgb importance analysis
    importance_xgb = importance.importance_xgb(best_model)
    filename = f"importance_xgb_seed_{seed}.csv"
    importance_xgb.to_csv(path / filename, index=False)
    logger.info("XGBoost importance metrics saved to '%s'", filename)

    # Perfom and save shap importance analysis
    importance_shap = importance.importance_shap(config_obj, best_model, X_test_shap)
    filename = f"importance_shap_seed_{seed}.pkl"
    joblib.dump(importance_shap, path / filename, compress=True)
    logger.info("Shap importance metrics saved to '%s'", filename)

    # Save predicted probabilities
    filename = f"y_cls_proba_test_seed_{seed}.parquet"
    pd.DataFrame(
        best_model.predict_proba(X_test),
        columns=[f"{y_test.columns[0]}_class_{i}" for i in range(config_obj.n_quantile)],
        index=y_test.index,
    ).to_parquet(path / filename)
    logger.info("Target test set predicted probabilities saved to '%s'", filename)


def _feature_selection_loop(
    config_obj: config.AnalysisConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    X_test_shap: pd.DataFrame,
    params,
    seed,
    path,
):
    # Initialize features sets
    ranking_features = list(X_train.columns)
    regression_features = []

    # Initialize residuals
    res_train = y_train.copy().iloc[:, 0]
    res_val = y_val.copy().iloc[:, 0]

    # Initialize stopping critieria params
    patience = config_obj.patience
    counter = 0
    tol = config_obj.tol

    # Others
    best_rmse = None
    best_model = None
    best_features = None
    history = []

    should_stop = False
    while not should_stop:
        # Train the ranking model
        ranking_model = XGBRegressor(random_state=seed, **params)
        ranking_model.fit(
            X_train[ranking_features],
            res_train,
            eval_set=[(X_val[ranking_features], res_val)],
            verbose=False,
        )

        # Extract the most informative feature
        importance_analysis = importance.importance_xgb(ranking_model)
        ranking = importance_analysis.sort_values(
            by=config_obj.xgb_importance,
            ascending=False,
        )
        best_feature = ranking.index[0]

        # Update features sets
        ranking_features.remove(best_feature)
        regression_features.append(best_feature)

        # Train the regressor model
        regression_model = XGBRegressor(random_state=seed, **params)
        regression_model.fit(
            X_train[regression_features],
            y_train.iloc[:, 0],
            eval_set=[(X_val[regression_features], y_val.iloc[:, 0])],
            verbose=False,
        )

        # Compute training residuals for the next ranking iteration
        y_train_pred = regression_model.predict(X_train[regression_features])
        res_train = y_train.iloc[:, 0] - y_train_pred

        # Compute validationresiduals for the next ranking iteration
        y_val_pred = regression_model.predict(X_val[regression_features])
        res_val = y_val.iloc[:, 0] - y_val_pred

        # Verify the stopping criteria
        current_rmse = root_mean_squared_error(y_val, y_val_pred)
        if best_rmse is None:
            best_rmse = current_rmse
            best_model = regression_model
            best_features = regression_features.copy()
            counter = 0
            history.append(
                {
                    "iteration": len(regression_features),
                    "selected_feature": best_feature,
                    "gain": ranking.loc[best_feature, config_obj.xgb_importance],
                    "validation_rmse": current_rmse,
                    "relative_improvement": np.nan,
                    "n_selected_features": len(regression_features),
                }
            )

            continue

        improvement = (best_rmse - current_rmse) / best_rmse

        history.append(
            {
                "iteration": len(regression_features),
                "selected_feature": best_feature,
                "gain": ranking.loc[best_feature, config_obj.xgb_importance],
                "validation_rmse": current_rmse,
                "relative_improvement": (np.nan if best_rmse is None else improvement),
                "n_selected_features": len(regression_features),
            }
        )

        if improvement > tol:
            best_rmse = current_rmse
            best_model = regression_model
            best_features = regression_features.copy()
            counter = 0
        else:
            counter += 1

        should_stop = (counter > patience) or (len(ranking_features) == 0)

    # Set parameters for final training
    final_params = params.copy()
    final_params.pop("early_stopping_rounds", None)
    assert isinstance(best_model, XGBRegressor)
    final_params["n_estimators"] = best_model.best_iteration + 1

    # Retrain best model
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    best_model_final = XGBRegressor(random_state=seed, **final_params)
    best_model_final.fit(X_train_val[best_features], y_train_val.iloc[:, 0], verbose=0)

    # Save best model
    assert isinstance(best_model_final, XGBRegressor)
    filename = f"model_seed_{seed}.ubj"
    best_model_final.save_model(path / filename)
    logger.info("Best model saved to '%s'", filename)

    # Save selected features
    assert isinstance(best_features, list)
    filename = f"selected_features_seed_{seed}.csv"
    pd.DataFrame(
        {
            "rank": np.arange(1, len(best_features) + 1),
            "feature": best_features,
        }
    ).to_csv(path / filename, index=False)
    logger.info("Selected features saved to '%s'", filename)

    # Save selection history
    filename = f"selection_history_seed_{seed}.csv"
    pd.DataFrame(history).to_csv(path / filename, index=False)
    logger.info("Selection history saved to '%s'", filename)

    # Evaluate on the test set
    y_test_pred = best_model_final.predict(X_test[best_features])

    filename = f"y_test_pred_seed_{seed}.parquet"
    pd.DataFrame(
        y_test_pred,
        columns=y_test.columns,
        index=y_test.index,
    ).to_parquet(path / filename)
    logger.info("Test predictions saved to '%s'", filename)

    # Perform and save xgb importance analysis
    importance_xgb = importance.importance_xgb(best_model_final)
    filename = f"importance_xgb_seed_{seed}.csv"
    importance_xgb.to_csv(path / filename, index=False)
    logger.info("XGBoost importance metrics saved to '%s'", filename)

    # Perform and save SHAP analysis
    importance_shap = importance.importance_shap(
        config_obj,
        best_model_final,
        X_test_shap[best_features],
    )
    filename = f"importance_shap_seed_{seed}.pkl"
    joblib.dump(
        importance_shap,
        path / filename,
        compress=True,
    )
    logger.info("SHAP importance saved to '%s'", filename)


def _save_config(config_obj: config.AnalysisConfig, path: Path, filename: str):
    with open(path / filename, "w") as f:
        yaml.safe_dump(asdict(config_obj), f, sort_keys=False)


def _save_paths(paths: data_paths.DataPathsConfig, path: Path, filename: str):
    with open(path / filename, "w") as f:
        k, v = paths.target
        paths_dict = {
            "target": {k: str(v)},
            "features": {k: str(v) for k, v in paths.features.items()},
        }
        yaml.safe_dump(paths_dict, f, sort_keys=False)
