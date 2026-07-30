import pandas as pd
from pandas.api.typing import SeriesGroupBy

from thesis_project.rl_trading.data import BASE_FEATURES
from thesis_project.rl_trading.dataset_config import DatasetConfig
from thesis_project.rl_trading.features import FeatureSpec


def _validate_feature_specs(feature_specs: list[FeatureSpec]):
    """
    Check for duplicates in FeatureSpec base ids
    """
    seen = set()
    for spec in feature_specs:
        if spec.base_id in seen:
            raise ValueError(f"Duplicate FeatureSpec object '{spec.base_id}'")

        seen.add(spec.base_id)

    seen = set()
    for spec in feature_specs:
        for name in spec.output_names:
            if name in seen:
                raise ValueError(f"Duplicate feature '{name}'")

            seen.add(name)


def _load_feature(spec: FeatureSpec) -> pd.Series:
    """
    Load a base feature and perform preliminary checks.
    """
    base = BASE_FEATURES[spec.base_id]
    df = pd.read_parquet(base.path)

    if df.empty:
        raise ValueError(f"BaseFeature '{base.base_id}': empty DataFrame")

    if df.shape[1] != 1:
        raise ValueError(
            f"BaseFeature '{base.base_id}': expected 1 column DataFrame, found {df.shape[1]} columns instead"
        )

    series = df.squeeze("columns")

    if not isinstance(series, pd.Series):
        raise ValueError(
            f"BaseFeature '{base.base_id}': expected a Series after squeeze, found {type(series)} instead"
        )

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(
            f"BaseFeature '{base.base_id}': expected a DatetimeIndex, found {type(series.index)} instead"
        )

    series = series.rename(base.base_id)

    return series


def _filter_time(idx: pd.DatetimeIndex, min_time: str, max_time: str) -> pd.DatetimeIndex:
    """
    Filter the index between min and max time. Time must be in hh:mm:ss format
    """
    mask = idx.indexer_between_time(min_time, max_time)

    return idx[mask]


def _filter_dates(idx: pd.DatetimeIndex, excluded: set[pd.Timestamp]) -> pd.DatetimeIndex:
    """ """
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex")

    dates = idx.normalize().tz_localize(None)
    mask = ~dates.isin(excluded)

    return idx[mask]


def _validate_series(s: pd.Series):
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex")

    if s.index.has_duplicates:
        raise ValueError("Duplicates found in the index")

    if s.index.hasnans:
        raise ValueError("NaNs found in the index")


def _validate_grouped(grouped: SeriesGroupBy):
    for _, day in grouped:
        diffs = day.index[1:] - day.index[:-1]
        if (diffs != pd.Timedelta(seconds=1)).any():
            raise ValueError("Gaps found in index")


def _build_spec(config: DatasetConfig, spec: FeatureSpec) -> dict[str, pd.Series]:
    """ """
    # Load and validate base feature
    s = _load_feature(spec)
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    _validate_series(s)

    # Filter dates
    assert isinstance(s.index, pd.DatetimeIndex)
    idx = _filter_dates(s.index, config.dates_to_exclude)

    # Filter time
    lookback = spec.max_lag + spec.max_lookback
    min_time, max_time = config.get_preprocessing_interval(lookback)
    idx = _filter_time(idx, min_time, max_time)

    # Filter base feature
    s = s[idx]
    if s.empty:
        raise ValueError("The Series is empty")

    # Create and validate group
    g = s.groupby(pd.Grouper("D"))
    _validate_grouped(g)

    # Compute transforms
    transformed = {}
    for transform in spec.transforms:
        transformed.update(transform.transform(s, g, spec.base_id))

    return transformed


def build_dataset(config: DatasetConfig, specs: list[FeatureSpec]) -> pd.DataFrame:
    _validate_feature_specs(specs)

    transformed = {}
    # TODO: Implement parallel version
    for spec in specs:
        transformed.update(_build_spec(config, spec))

    df = pd.DataFrame(transformed)
    df = df.between_time(config.min_time, config.max_time)

    if df.empty:
        raise ValueError("The DataFrame is empty")

    return df
