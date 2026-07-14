"""Orchestration pipeline for feature engineering analysis."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from thesis_project import global_config as global_config

from .config import AnalysisConfig
from .data_paths import DataPathsConfig, get_data_paths
from .data_pipeline import prepare_analysis_data
from .metrics import compute_metrics
from .sampling import sample_data

logger = logging.getLogger(__name__)

_DATA = None


def validate_data_structure(data: pd.DataFrame) -> pd.DataFrame:
    """Validate timestamp index structure and ordering."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be DatetimeIndex, got {type(data.index).__name__}")

    if not data.index.is_monotonic_increasing:
        logger.warning("Data index not monotonic increasing, sorting...")
        data = data.sort_index()

    # Check for duplicate timestamps
    if len(data.index) != len(data.index.unique()):
        duplicates = data.index[data.index.duplicated()]
        raise ValueError(
            f"Found {len(duplicates)} duplicate timestamps in index. "
            f"First few: {duplicates[:5].tolist()}"
        )

    return data


def generate_run_name(config: AnalysisConfig) -> int:
    """Generate next incremental run number."""
    if config.output_path is None:
        raise ValueError("config.output_path must be set")

    config.output_path.mkdir(parents=True, exist_ok=True)

    existing_runs = []
    for item in config.output_path.iterdir():
        if item.is_dir():
            try:
                item_num = item.name.split("_")[-1]
                existing_runs.append(int(item_num))
            except ValueError:
                pass

    return max(existing_runs) + 1 if existing_runs else 1


def save_config_snapshot(
    config: AnalysisConfig,
    run_dir: Path,
    run_num: int,
    data_paths: DataPathsConfig,
) -> None:
    """Save config and data paths to JSON for reproducibility."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Convert config to dict
    config_dict = {
        "ticker": config.ticker,
        "metrics": config.metrics,
        "analysis_horizons": config.analysis_horizons,
        "sample_ratio": config.sample_ratio,
        "sampling_strategy": config.sampling_strategy,
        "metrics_params": config.metrics_params,
        "seed": config.seed,
        "n_jobs": config.n_jobs,
        "min_time": config.min_time,
        "max_time": config.max_time,
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_paths": {
            "ticker": data_paths.ticker,
            "targets": {k: str(v) for k, v in data_paths.targets.items()},
            "features": {k: str(v) for k, v in data_paths.features.items()},
        },
    }

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Configuration settings saved: %s", config_path.relative_to(global_config.ROOT))


def _compute_tuple_metrics(
    horizon: int,
    target_col: str,
    feature_col: str,
    config: AnalysisConfig,
) -> Dict:
    """Compute metrics for (horizon, target, feature) tuple."""
    try:
        # Access global data
        global _DATA
        if _DATA is not None:
            target_series = _DATA[target_col]
            feature_series = _DATA[feature_col]
        else:
            raise ValueError("Data not loaded in worker process")

        # Validate inputs
        if len(feature_series) == 0 or len(target_series) == 0:
            return {
                "horizon": horizon,
                "target": target_col,
                "feature": feature_col,
                "n_samples": 0,
                "timestamp": pd.Timestamp.now(),
                "error": "Empty input series",
            }

        # Align by index
        aligned = pd.concat(
            [feature_series, target_series],
            axis=1,
        )
        aligned.columns = ["x", "z"]

        assert isinstance(aligned.index, pd.DatetimeIndex)
        aligned["y"] = aligned.groupby(aligned.index.date)["z"].shift(-horizon)

        # Drop NaNs
        aligned = aligned.dropna()

        if len(aligned) < 10:
            logger.info(
                "Insufficient valid samples after alignment: %d for h=%d, target=%s, feature=%s",
                len(aligned),
                horizon,
                target_col,
                feature_col,
            )
            return {
                "horizon": horizon,
                "target": target_col,
                "feature": feature_col,
                "n_samples": len(aligned),
                "timestamp": pd.Timestamp.now(),
                "error": f"Insufficient valid samples after alignment: {len(aligned)} < 10",
            }

        # Extract aligned arrays
        x = aligned["x"].values.astype(np.float64)
        y = aligned["y"].values.astype(np.float64)
        z = aligned["z"].values.astype(np.float64)

        # Sampling if configured
        if config.sampling_strategy and config.sample_ratio > 0:
            df_sample = pd.DataFrame(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
            df_sampled = sample_data(
                df_sample,
                config.sampling_strategy,
                {"sample_ratio": config.sample_ratio, "random_state": config.seed},
            )
            x_final = df_sampled["x"].values.astype(np.float64)
            y_final = df_sampled["y"].values.astype(np.float64)
            z_final = df_sampled["z"].values.astype(np.float64)
        else:
            x_final = x
            y_final = y
            z_final = z

        # Ensure contiguous memory
        x_final = np.ascontiguousarray(x_final, dtype=np.float64)
        y_final = np.ascontiguousarray(y_final, dtype=np.float64)
        z_final = np.ascontiguousarray(z_final, dtype=np.float64)

        # Compute metrics
        metrics_dict = compute_metrics(
            x_final,
            y_final,
            z_final,
            config.metrics,
            config.metrics_params,
        )

        # Build result
        result = {
            "horizon": horizon,
            "target": target_col,
            "feature": feature_col,
            "n_samples": len(x_final),
            "timestamp": pd.Timestamp.now(),
        }

        # Add metric values
        for metric in config.metrics:
            result[metric] = metrics_dict.get(metric, np.nan)

        return result

    except Exception as e:
        logger.exception(
            "Computation failed: h=%d, target=%s, feature=%s",
            horizon,
            target_col,
            feature_col,
        )
        return {
            "horizon": horizon,
            "target": target_col,
            "feature": feature_col,
            "n_samples": 0,
            "timestamp": pd.Timestamp.now(),
            "error": str(e),
        }


def run_analysis(config: AnalysisConfig) -> pd.DataFrame:
    """
    Run complete feature engineering analysis.

    Pipeline:
    - Load data
    - Merge on index
    - Validate/sort
    - Generate tasks
    - Parallel workers (local session-safe shifting)
    - Collect results
    - Save single parquet

    Args:
        config: AnalysisConfig with all parameters

    Returns:
        Tuple of (results_dataframe, metadata_dict)
    """
    # Validate config
    if not config.ticker:
        raise ValueError("config.ticker must be set")

    logger.info(
        f"""

        Analysis Configuration:

            Ticker: {config.ticker}
            Metrics: {", ".join(config.metrics)}
            Sample ratio: {config.sample_ratio * 100:.2f}%
            Horizons: {", ".join(map(str, config.analysis_horizons))}
            Parallel jobs: {config.n_jobs}

        """
    )

    # Get data paths
    data_paths = get_data_paths(config.ticker)

    # Generate run number
    run_num = generate_run_name(config)
    assert config.output_path is not None
    run_dir = config.output_path / f"run_{run_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler to log to run-specific file
    log_file = run_dir / f"run_{run_num:03d}.log"
    file_handler = logging.FileHandler(log_file)
    fmt = logging.Formatter(
        "%(asctime)-23s | %(levelname)-8s | %(filename)-20s:%(lineno)-4d | %(message)s"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("Run number: %s", run_num)
    logger.info("Output directory: %s", run_dir.relative_to(global_config.ROOT))
    logger.info("Logging to: %s", log_file.relative_to(global_config.ROOT))

    # Save config snapshot with data paths
    save_config_snapshot(config, run_dir, run_num, data_paths)

    # Load data
    targets_df, features_df = prepare_analysis_data(config, config.ticker)

    # Merge on index (index-based, not column-based)
    print()
    logger.info("Merging targets and features")
    data = pd.merge(
        targets_df,
        features_df,
        left_index=True,
        right_index=True,
        how="outer",
    )
    logger.info("Data merged, shape: %s", data.shape)
    data = validate_data_structure(data)

    global _DATA
    _DATA = data

    # Extract column names
    target_cols = list(targets_df.columns)
    feature_cols = list(features_df.columns)

    if not target_cols:
        raise ValueError("No target columns found in targets_df")
    if not feature_cols:
        raise ValueError("No feature columns found in features_df")

    logger.info(
        "Targets (total: %d) (size: %.2f MB):",
        len(target_cols),
        data[target_cols].memory_usage(deep=True).sum() / 1e6,
    )
    for col in target_cols:
        logger.info("  - %s (size: %.2f MB)", col, data[col].memory_usage(deep=True) / 1e6)
    logger.info(
        "Features (total: %d) (size: %.2f MB):",
        len(feature_cols),
        data[feature_cols].memory_usage(deep=True).sum() / 1e6,
    )
    for col in feature_cols:
        logger.info("  - %s (size: %.2f MB)", col, data[col].memory_usage(deep=True) / 1e6)

    # Generate tasks
    print()
    logger.info("Generating tuples (horizon, target_col, feature_col)")
    tasks = [
        (horizon, target_col, feature_col)
        for horizon in config.analysis_horizons
        for target_col in target_cols
        for feature_col in feature_cols
    ]

    logger.info(
        "Tuples generated: %d (%d horizons x %d targets x %d features)",
        len(tasks),
        len(config.analysis_horizons),
        len(target_cols),
        len(feature_cols),
    )

    start_time = time.time()

    print()
    if config.n_jobs == 1:
        logger.info("Running in serial mode (n_jobs=%s)", config.n_jobs)
        results = []
        for h, t, f in tqdm(tasks, desc="Computing metrics", unit="task"):
            result = _compute_tuple_metrics(
                h,
                t,
                f,
                config,
            )
            results.append(result)
    else:
        logger.info("Running in parallel mode (n_jobs=%s)", config.n_jobs)
        with tqdm_joblib(
            total=len(tasks),
            desc="Computing metrics",
            unit="tuple",
            dynamic_ncols=False,
            leave=True,
            position=0,
            ascii=True,
        ):
            results = Parallel(
                n_jobs=config.n_jobs,
                backend="threading",
                # max_nbytes="1M",
                # mmap_mode="r",
                # verbose=10,
            )(
                delayed(_compute_tuple_metrics)(
                    h,
                    t,
                    f,
                    config,
                )
                for h, t, f in tasks
            )

    assert isinstance(results, list)
    results_df = pd.DataFrame(results)
    results_path = run_dir / "results.parquet"
    os.makedirs(run_dir, exist_ok=True)
    results_df.to_parquet(results_path, index=False)

    print()
    logger.info(
        f"""

        Analysis completed:

            Time elapsed: {time.time() - start_time:.2f} seconds.
            Results shape: {results_df.shape}
            Saved to: {results_path.relative_to(global_config.ROOT)}

        """
    )

    return results_df
