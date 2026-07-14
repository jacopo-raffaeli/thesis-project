import datetime
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from thesis_project import config as global_config
from thesis_project.importance_xgb import (
    config,
    data_paths,
    registry,
    sampling,
    temporal,
)

logger = logging.getLogger(__name__)


def get_min_time_offset(config_obj: config.AnalysisConfig) -> int:
    """
    Get the cutoff offset from start of day in seconds

    # Args:
        - config_obj: Analysis configuration file

    # Return:
        - Offset
    """

    # Initialize to avoid bounds errors
    max_delta = 0
    max_window = 0

    for feature_transform in config_obj.feature_transforms:
        if feature_transform.name == "delta":
            deltas = feature_transform.params["deltas"]
            max_delta = max(deltas) if deltas is not None else 0

        if feature_transform.name == "rolling":
            windows = feature_transform.params["windows"]
            max_window = max(windows) if windows is not None else 0

    offset = max(max_delta, max_window)

    return offset


def get_max_time_offset(config_obj: config.AnalysisConfig) -> int:
    """
    Get the cutoff offset from end of day in seconds

    # Args:
        - config_obj: Analysis configuration object

    # Return:
        - Offset
    """

    offset = config_obj.horizon

    return offset


def filter_time(
    idx: pd.DatetimeIndex,
    min_time: str = "09:00:00",
    max_time: str = "17:00:00",
) -> pd.DatetimeIndex:
    """
    Filter DataFrame by time of day with optional buffer.

    # Args:
        - idx: Index to filter
        - min_time: Minimum time to keep
        - max_time: Maximum time to keep
        - buffer_seconds: Additional buffer to substract to min_time

    # Return:
        - Filtered index
    """
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a pd.DatetimeIndex")

    min_dt = pd.to_datetime(min_time)
    min_time_obj = min_dt.time()
    max_dt = pd.to_datetime(max_time)
    max_time_obj = max_dt.time()
    mask = (idx.time >= min_time_obj) & (idx.time <= max_time_obj)

    return idx[mask]


def filter_dates(
    idx: pd.DatetimeIndex,
    ticker: str,
    date_filters_offsets: Dict[str, List[int]],
) -> pd.DatetimeIndex:
    """
    Filter DataFrame by structural dates and futures last trading days.

    # Args:
        - idx: Index to filter
        - ticker: Asset ticker, either 'fbtp' or 'fbts'
        - date_filters_offset: Dictionary containing an identifier and the number of days to filter before and after dates of such category

    # Return:
        - Filtered index
    """
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    if ticker not in global_config.DATES_TO_EXCLUDE:
        raise ValueError(f"{ticker} not found")

    dates_to_exclude_dict = global_config.DATES_TO_EXCLUDE[ticker]

    # Get boundary dates
    min_date = idx.min().date()
    max_date = idx.max().date()

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

    dates_index = pd.Index(idx.date)
    present_dates = set(dates_index.unique())
    matched_dates = present_dates & dates_to_filter
    logger.info("Dates to filter: %d", len(matched_dates))

    if not dates_to_filter:
        return idx

    logger.info("Dates before filtering: %d", dates_index.nunique())
    mask = ~dates_index.isin(dates_to_filter)
    idx = idx[mask]

    assert isinstance(idx, pd.DatetimeIndex)
    logger.info("Dates after filtering: %d", pd.Index(idx.date).nunique())

    return idx


def load_data(name: str, path: Path) -> pd.DataFrame:
    """
    Load a single column pd.DataFrame and execute preliminary checks

    # Args:
        - name: String used to rename the column
        - path: Path object to the file

    # Return:
        - Pandas dataframe with the series
    """

    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")

    if not path.is_file():
        raise ValueError(f"Path {path} is not a file")

    logger.debug("Loading '%s' from '%s'", name, path.name)
    df = pd.read_parquet(path)

    if len(df.columns) != 1:
        raise ValueError(f"File {path} has more than one column")

    assert isinstance(df.index, pd.DatetimeIndex)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    df = df.rename(columns={df.columns[0]: name})
    return df


def preprocess_timestamps(
    config_obj: config.AnalysisConfig, idx: pd.DatetimeIndex
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Preprocess timestamp series. Filter days and time, sample subset

    # Args:
        - config_obj: Analysis configuration object
        - idx: pd.DatetimeIndex to process

    # Return:
        - Sampled timestamps series
    """
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Index must be a pd.DateTimeIndex")

    # Find min valid sample time
    min_time = config_obj.min_time
    min_time_offset = get_min_time_offset(config_obj)
    min_time_sample = pd.Timedelta(min_time) + pd.Timedelta(min_time_offset, unit="seconds")
    min_time_sample = str(min_time_sample).split()[-1]

    # Find max valid sample time
    max_time = config_obj.max_time
    max_time_offset = get_max_time_offset(config_obj)
    max_time_sample = pd.Timedelta(max_time) - pd.Timedelta(max_time_offset, unit="seconds")
    max_time_sample = str(max_time_sample).split()[-1]

    # Filter timestamp series
    idx_filtered = filter_time(idx, min_time=min_time_sample, max_time=max_time_sample)
    idx_filtered = filter_dates(idx_filtered, config_obj.ticker, config_obj.date_filters_offsets)

    # Sample timestamps
    strategy = config_obj.sampling_strategy
    sampling_params = config_obj.sampling_params
    idx_sampled = sampling.sample_timestamps(idx_filtered, strategy, sampling_params)
    idx_sampled = idx_sampled.sort_values()
    dates_sampled = idx_sampled.normalize().unique()

    return idx_sampled, dates_sampled


def build_transforms(
    transform_specs: List[config.TransformSpec] | config.TransformSpec, columns: str
) -> List:
    if isinstance(transform_specs, config.TransformSpec):
        transform_specs = [transform_specs]

    transforms = []
    for spec in transform_specs:
        transform_cls = registry.TRANSFORM_REGISTRY[spec.name]
        transform = transform_cls(columns=[columns], **spec.params)
        transforms.append(transform)

    return transforms


def preprocess_target(
    config_obj: config.AnalysisConfig,
    target_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
    """ """
    assert isinstance(target_df.index, pd.DatetimeIndex)
    target_idx_norm = target_df.index.normalize()
    grouped = target_df.groupby(target_idx_norm)

    # Search target transform if enabled
    if not config_obj.use_base_target:
        columns = target_df.columns[0]
        transform = build_transforms(config_obj.target_transform, columns)[0]
        target_df = transform.compute(target_df, grouped)
        assert isinstance(target_df.index, pd.DatetimeIndex)
        target_idx_norm = target_df.index.normalize()
        grouped = target_df.groupby(target_idx_norm)

    target_shifted_df = grouped.shift(-config_obj.horizon)
    mask = target_shifted_df.notna().all(axis=1)
    target_shifted_df = target_shifted_df[mask]

    assert isinstance(target_shifted_df.index, pd.DatetimeIndex)
    idx_sampled, dates_sampled = preprocess_timestamps(config_obj, target_shifted_df.index)
    y = target_shifted_df.loc[idx_sampled]

    return y, idx_sampled, dates_sampled


def preprocess_market_feature(
    config_obj: config.AnalysisConfig,
    feat_name: str,
    feat_path: Path,
    idx_sampled: pd.DatetimeIndex,
    dates_sampled: pd.DatetimeIndex,
) -> pd.DataFrame:
    feat_df = load_data(feat_name, feat_path)

    assert isinstance(feat_df.index, pd.DatetimeIndex)
    feat_idx_norm = feat_df.index.normalize()
    mask = feat_idx_norm.isin(dates_sampled)

    feat_df = feat_df.loc[mask]
    feat_idx_norm = feat_idx_norm[mask]

    grouped_by_session = feat_df.groupby(feat_idx_norm)
    transforms = build_transforms(config_obj.feature_transforms, feat_name)

    derived_blocks = []
    for transform in transforms:
        derived_blocks.append(transform.compute(feat_df, grouped_by_session))

    if derived_blocks:
        derived_df = pd.concat(
            derived_blocks,
            axis=1,
        )

        if not feat_df.index.equals(derived_df.index):
            raise ValueError(
                f"Indexes of base and derived features do not match for feature '{feat_name}'"
            )

        feat_df = feat_df.join(derived_df)

    feat_df = feat_df.loc[idx_sampled]

    return feat_df


def preprocess_market_features(
    config_obj: config.AnalysisConfig,
    idx_sampled: pd.DatetimeIndex,
    dates_sampled: pd.DatetimeIndex,
) -> pd.DataFrame:
    paths = data_paths.get_data_paths(config_obj.ticker)
    market_features = Parallel(
        n_jobs=config_obj.n_jobs_preprocessing,
        backend="loky",
        verbose=10,
    )(
        delayed(preprocess_market_feature)(
            config_obj=config_obj,
            feat_name=feat_name,
            feat_path=feat_path,
            idx_sampled=idx_sampled,
            dates_sampled=dates_sampled,
        )
        for feat_name, feat_path in paths.features.items()
    )

    market_features_df = pd.concat(
        market_features,
        axis=1,
    )

    return market_features_df


def preprocess_temporal_features(
    config_obj: config.AnalysisConfig,
    idx_sampled: pd.DatetimeIndex,
    dates_sampled: pd.DatetimeIndex,
) -> pd.DataFrame:
    temporal_features_df = pd.DataFrame(index=idx_sampled)

    # Plain calendar features
    for name in config_obj.calendar_features:
        calendar_feature_fn = registry.CALENDAR_FEATURES_REGISTRY[name]
        temporal_features_df[name] = calendar_feature_fn(idx_sampled)

    # Encoded calendard features
    for name in config_obj.calendar_features_encoded:
        calendar_feature_fn = registry.CALENDAR_FEATURES_ENCODED_REGISTRY[name]
        sin, cos = calendar_feature_fn(idx_sampled)
        temporal_features_df[f"{name}_sin"] = sin
        temporal_features_df[f"{name}_cos"] = cos

    # Event based features
    base_path = global_config.RAW_DIR / config_obj.ticker
    for name, metadata in config_obj.event_based_features.items():
        path = base_path / metadata["file"]
        column = metadata["column"]
        dates = pd.read_csv(path, parse_dates=[column])[column]

        for transform_key in metadata["transforms"]:
            transform = registry.EVENT_BASED_FEATURES_REGISTRY[transform_key]
            temporal_features_df[f"{transform_key}_{name}"] = transform(idx_sampled, dates)

    # Daily event based features
    for name, metadata in config_obj.daily_event_based_features.items():
        for unit in metadata.get("units", []):
            temporal_features_df[f"{name}_{unit}"] = temporal.time_to_daily_event(
                idx_sampled,
                metadata["event_time"],
                metadata["event_tz"],
                unit,
            )

    return temporal_features_df


def preprocess_features(
    config_obj: config.AnalysisConfig,
    idx_sampled: pd.DatetimeIndex,
    dates_sampled: pd.DatetimeIndex,
) -> pd.DataFrame:
    logger.info("Preprocessing market features")
    mrkt_df = preprocess_market_features(config_obj, idx_sampled, dates_sampled)
    logger.info("Market features preprocessed")
    logger.debug("List of market features:")
    for col in mrkt_df.columns:
        data = mrkt_df[col]
        n_data = len(data)
        n_nan = data.isna().sum(axis=0)
        perc_nan = data.isna().mean() * 100
        logger.debug("  - '%s': samples = %d, nans = %d (%.2f%%)", col, n_data, n_nan, perc_nan)

    logger.info("Preprocessing temporal features")
    temp_df = preprocess_temporal_features(config_obj, idx_sampled, dates_sampled)
    logger.info("Temporal features preprocessed")
    logger.debug("List of temporal features:")
    for col in temp_df.columns:
        data = temp_df[col]
        n_data = len(data)
        n_nan = data.isna().sum(axis=0)
        perc_nan = data.isna().mean() * 100
        logger.debug("  - '%s': samples = %d, nans = %d (%.2f%%)", col, n_data, n_nan, perc_nan)

    if not temp_df.index.equals(mrkt_df.index):
        raise ValueError("Indexes of market and temporal features dataframes do not match")

    feat_df = temp_df.join(mrkt_df)
    logger.info("Market and temporal features joined")

    return feat_df


def match_X_y(
    config_obj: config.AnalysisConfig, X: pd.DataFrame, y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not X.index.equals(y.index):
        raise ValueError("Indexes of features and target dataframes do not match")

    # Drop nans
    mask = y.notna().all(axis=1)
    if not config_obj.keep_features_nan:
        mask &= X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    return X, y


def split_data(
    config_obj: config.AnalysisConfig, X: pd.DataFrame, y: pd.DataFrame
) -> Dict[str, Dict[str, pd.DataFrame]]:
    idx = y.index
    assert isinstance(idx, pd.DatetimeIndex)
    idx_norm = idx.normalize()
    dates = idx_norm.unique()
    n_dates = len(dates)

    perc_train = config_obj.perc_train
    perc_val = config_obj.perc_val

    end_train = int(n_dates * perc_train)
    end_val = int(n_dates * (perc_train + perc_val))

    dates_train = dates[:end_train]
    dates_val = dates[end_train:end_val]
    dates_test = dates[end_val:]

    if len(dates_train) == 0:
        raise ValueError("Dates list for training is empty")
    if len(dates_val) == 0:
        raise ValueError("Dates list for validation is empty")
    if len(dates_test) == 0:
        raise ValueError("Dates list for testing is empty")

    mask_train = idx_norm.isin(dates_train)
    mask_val = idx_norm.isin(dates_val)
    mask_test = idx_norm.isin(dates_test)

    data_dict = {
        "train": {"features": X[mask_train], "target": y[mask_train]},
        "validation": {"features": X[mask_val], "target": y[mask_val]},
        "test": {"features": X[mask_test], "target": y[mask_test]},
    }

    return data_dict


def compute_bins(y: pd.Series, n_quantile: int):
    # Compute bins
    _, bins = pd.qcut(y, n_quantile, labels=False, retbins=True, duplicates="raise")

    # Extend bins
    bins[0] = -np.inf
    bins[-1] = np.inf

    return bins


def classify_target(y: pd.DataFrame, bins) -> pd.DataFrame:
    col = y.columns[0]
    y_cls = pd.cut(y[col], bins, labels=False, include_lowest=True, duplicates="raise")

    return pd.DataFrame({col: y_cls}, index=y.index)
