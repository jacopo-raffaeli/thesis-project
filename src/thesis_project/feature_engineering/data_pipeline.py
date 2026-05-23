"""Data loading and preprocessing functions for feature engineering analysis."""

import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from thesis_project import config
from thesis_project.feature_engineering.config import AnalysisConfig
from thesis_project.feature_engineering.transforms import DeltaTransform, RollingTransform

logger = logging.getLogger(__name__)


def load_data(data_dict: Dict[str, Path]) -> pd.DataFrame:
    """Load and merge multiple parquet files on index."""
    dfs = []
    for name, path in data_dict.items():
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        if not path.is_file():
            raise ValueError(f"Path {path} is not a file")

        logger.info("Loading '%s' from '%s'", name, path.name)
        df = pd.read_parquet(path)
        df = df.rename(columns={df.columns[0]: name})
        dfs.append(df)

    if len(dfs) == 1:
        logger.info("Merging %d dataframe", len(dfs))
    elif len(dfs) > 1:
        logger.info("Merging %d dataframes", len(dfs))
    return pd.concat(dfs, axis=1, join="outer")


def filter_time(
    df: pd.DataFrame,
    min_time: str = "09:00:00",
    max_time: str = "17:00:00",
    buffer_seconds: int = 0,
) -> pd.DataFrame:
    """Filter DataFrame by time of day (with optional)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    min_dt = pd.to_datetime(min_time)
    if buffer_seconds > 0:
        min_dt = min_dt - datetime.timedelta(seconds=buffer_seconds)
    min_time_obj = min_dt.time()

    max_dt = pd.to_datetime(max_time)
    max_time_obj = max_dt.time()

    return df.between_time(min_time_obj, max_time_obj)


def filter_dates(
    df: pd.DataFrame,
    ticker: str,
    date_filters_offsets: Dict[str, List[int]],
) -> pd.DataFrame:
    """Filter DataFrame by structural dates and futures last trading days."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    if ticker not in config.DATES_TO_EXCLUDE:
        raise ValueError(f"{ticker=} not found")

    dates_to_exclude_dict = config.DATES_TO_EXCLUDE[ticker]

    # Get boundary dates
    min_date = df.index.min().date()
    max_date = df.index.max().date()

    dates_to_filter = set()

    # Structural dates
    logger.info("Collecting structural dates to filter")
    for d in dates_to_exclude_dict.get("structural", []):
        d = pd.Timestamp(d).date()

        if min_date <= d <= max_date:
            dates_to_filter.add(d)

    # Offset-expanded futures dates
    logger.info("Collecting futures last trading days to filter with offsets")
    if "fut_last_trading_days" in date_filters_offsets:
        before, after = date_filters_offsets["fut_last_trading_days"]

        for d in dates_to_exclude_dict.get("fut_last_trading_days", []):
            d = pd.Timestamp(d).date()
            start = d - datetime.timedelta(days=before)
            end = d + datetime.timedelta(days=after)
            if end < min_date or start > max_date:
                continue
            cur = max(start, min_date)
            end = min(end, max_date)
            while cur <= end:
                dates_to_filter.add(cur)
                cur += datetime.timedelta(days=1)

    dates_index = pd.Index(df.index.date)
    present_dates = set(dates_index.unique())
    matched_dates = present_dates & dates_to_filter
    logger.info("Dates to filter: %d", len(matched_dates))

    if not dates_to_filter:
        return df

    logger.info("Dates before filtering: %d", dates_index.nunique())
    mask = ~dates_index.isin(dates_to_filter)
    df = df.loc[mask]

    assert isinstance(df.index, pd.DatetimeIndex)
    logger.info("Dates after filtering: %d", pd.Index(df.index.date).nunique())

    return df


def preprocess_series(
    data_dict: Dict[str, Path],
    ticker: str,
    transform_configs: Dict[str, Dict],
    min_time: str = "09:00:00",
    max_time: str = "17:00:00",
    drop_base: bool = False,
    date_filters_offsets: Optional[Dict[str, List[int]]] = None,
) -> pd.DataFrame:
    """Apply transforms, filters, and feature engineering to series data."""
    df = load_data(data_dict)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    assert isinstance(date_filters_offsets, dict)
    logger.info("Filtering dates")
    df = filter_dates(df, ticker, date_filters_offsets)

    # Calculate buffer for initial time filter
    columns = list(data_dict.keys())

    # Determine buffer from rolling windows in transforms
    buffer_seconds = 0
    for transform_name, params in transform_configs.items():
        if transform_name == "delta" and "time_deltas" in params:
            buffer_seconds = max(buffer_seconds, max(params["time_deltas"]))
        elif transform_name == "rolling" and "windows" in params:
            buffer_seconds = max(buffer_seconds, max(params["windows"]))

    logger.info(
        "Filtering time between %s and %s with buffer of %d seconds",
        min_time,
        max_time,
        buffer_seconds,
    )
    df = filter_time(df, min_time, max_time, buffer_seconds)

    # Compute session grouping once
    assert isinstance(df.index, pd.DatetimeIndex)
    sessions = df.index.normalize()
    grouped_by_session = df.groupby(sessions)

    # Set up transforms based on config
    transforms = []
    if "delta" in transform_configs and transform_configs["delta"]:
        time_deltas = transform_configs["delta"].get("time_deltas", [])
        if time_deltas:
            transforms.append(DeltaTransform(columns, time_deltas))

    if "rolling" in transform_configs and transform_configs["rolling"]:
        windows = transform_configs["rolling"].get("windows", [])
        stats = transform_configs["rolling"].get("stats", ["mean", "std"])
        if windows:
            transforms.append(RollingTransform(columns, windows, stats))

    # Compute features from transforms
    feature_blocks = []
    for transform in transforms:
        logger.info("Computing %s:", transform.__class__.__name__)
        feature_block = transform.compute(df, grouped_by_session)
        feature_blocks.append(feature_block)

    if feature_blocks:
        features_df = pd.concat(feature_blocks, axis=1)
        df = pd.concat([df, features_df], axis=1)

    logger.info("Filtering time between %s and %s with buffer of %d seconds", min_time, max_time, 0)
    df = filter_time(df, min_time, max_time, buffer_seconds=0)

    if drop_base:
        logger.info("Dropping base columns: %s", columns)
        df = df.drop(columns=columns, errors="ignore")

    return df


def prepare_analysis_data(
    config_obj: AnalysisConfig,
    ticker: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare targets and features DataFrames for analysis."""
    from thesis_project.feature_engineering.data_paths import get_data_paths

    # Get data paths for the ticker
    data_paths = get_data_paths(ticker)

    # Preprocess targets with target-specific transforms
    print()
    logger.info("Preprocessing target data")
    targets_df = preprocess_series(
        data_paths.targets,
        ticker,
        transform_configs=config_obj.target_transforms,
        min_time=config_obj.min_time,
        max_time=config_obj.max_time,
        drop_base=config_obj.drop_base_targets,
        date_filters_offsets=config_obj.date_filters_offsets,
    )
    logger.info("Preprocessing target data completed, data shape: %s", targets_df.shape)

    # Preprocess features with feature-specific transforms
    print()
    logger.info("Preprocessing feature data")
    features_df = preprocess_series(
        data_paths.features,
        ticker,
        transform_configs=config_obj.feature_transforms,
        min_time=config_obj.min_time,
        max_time=config_obj.max_time,
        drop_base=config_obj.drop_base_features,
        date_filters_offsets=config_obj.date_filters_offsets,
    )
    logger.info("Preprocessing feature data completed, data shape: %s", features_df.shape)

    return targets_df, features_df
