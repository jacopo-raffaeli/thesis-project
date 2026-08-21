import pandas as pd
from pandas.api.typing import SeriesGroupBy

from thesis_project.dataset.data import BASE_FEATURES, BaseFeature
from thesis_project.dataset.features import FeatureSpec
from thesis_project.rl_trading.dataset_config import DatasetConfig
from thesis_project.rl_trading.env import RLDataset
from thesis_project.rl_trading.env_config import EnvConfig


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


def _load_feature(base: BaseFeature) -> pd.Series:
    """
    Load a base feature and perform preliminary checks.
    """
    df = pd.read_parquet(base.path)

    if df.empty:
        raise ValueError(f"BaseFeature '{base.base_id}': empty DataFrame")

    if df.shape[1] != 1:
        raise ValueError(
            f"BaseFeature '{base.base_id}': expected 1 column DataFrame, found {df.shape[1]} columns instead"
        )

    series = df.squeeze("columns")
    assert isinstance(series, pd.Series)
    _validate_series(series)

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

    dates = idx.floor("D").tz_localize(None)
    mask = ~dates.isin(excluded)

    return idx[mask]


def _validate_series(s: pd.Series):
    if not isinstance(s, pd.Series):
        raise ValueError("Object must be a Series")

    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex")

    if s.index.has_duplicates:
        raise ValueError("Duplicates found in the index")

    if s.index.hasnans:
        raise ValueError("NaNs found in the index")


def _validate_grouped(grouped: SeriesGroupBy):
    for _, day in grouped:
        if day.empty:
            raise ValueError("Day is empty")

        diffs = day.index[1:] - day.index[:-1]
        if (diffs != pd.Timedelta(seconds=1)).any():
            raise ValueError("Gaps found in index")


def build_spec(config: DatasetConfig, spec: FeatureSpec) -> dict[str, pd.Series]:
    """ """
    # Load and validate base feature
    s = _load_feature(BASE_FEATURES[spec.base_id])
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()

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
    assert isinstance(s.index, pd.DatetimeIndex)
    g = s.groupby(s.index.floor(freq="D"))
    _validate_grouped(g)

    # Compute transforms
    transformed = {}
    for transform in spec.transforms:
        transformed.update(transform.transform(s, g, spec.base_id))

    return transformed


def build_spec_pl(config: DatasetConfig, spec: FeatureSpec) -> dict[str, pd.Series]:
    from datetime import time

    import polars as pl

    excluded = [d.date() for d in config.dates_to_exclude]
    lookback = spec.max_lag + spec.max_lookback
    min_time, max_time = config.get_preprocessing_interval(lookback)
    min_time = time.fromisoformat(min_time)
    max_time = time.fromisoformat(max_time)
    base = BASE_FEATURES[spec.base_id]

    df = (
        pl.scan_parquet(base.path)
        .filter(~pl.col("timestamp").dt.date().is_in(excluded))
        .filter(pl.col("timestamp").dt.time().is_between(min_time, max_time))
        .collect()
    )

    df = df.to_pandas()
    df = df.set_index("timestamp")

    if df.empty:
        raise ValueError(f"BaseFeature '{base.base_id}': empty DataFrame")

    if df.shape[1] != 1:
        raise ValueError(
            f"BaseFeature '{base.base_id}': expected 1 column DataFrame, found {df.shape[1]} columns instead"
        )

    s = df.squeeze("columns")
    assert isinstance(s, pd.Series)
    _validate_series(s)

    s = s.rename(base.base_id)

    if not s.index.is_monotonic_increasing:
        s = s.sort_index()

    if s.empty:
        raise ValueError("The Series is empty")

    # Create and validate group
    assert isinstance(s, pd.Series)
    assert isinstance(s.index, pd.DatetimeIndex)
    g = s.groupby(s.index.floor(freq="D"))
    _validate_grouped(g)

    # Compute transforms
    transformed = {}
    for transform in spec.transforms:
        transformed.update(transform.transform(s, g, spec.base_id))

    return transformed


def build_features_dataset(config: DatasetConfig, specs: list[FeatureSpec]) -> pd.DataFrame:
    _validate_feature_specs(specs)

    transformed = {}
    # TODO: Implement parallel version
    for spec in specs:
        transformed.update(build_spec(config, spec))

    df = pd.DataFrame(transformed)
    df = df.between_time(config.min_time, config.max_time)

    if df.empty:
        raise ValueError("The DataFrame is empty")

    return df


def build_features_dataset_pl(config: DatasetConfig, specs: list[FeatureSpec]) -> pd.DataFrame:
    _validate_feature_specs(specs)

    transformed = {}
    # TODO: Implement parallel version
    for spec in specs:
        transformed.update(build_spec_pl(config, spec))

    df = pd.DataFrame(transformed)
    df = df.between_time(config.min_time, config.max_time)

    if df.empty:
        raise ValueError("The DataFrame is empty")

    return df


def build_rl_dataset(
    dataset_config: DatasetConfig, env_config: EnvConfig, specs: list[FeatureSpec]
) -> RLDataset:
    features = build_features_dataset(dataset_config, specs)
    assert isinstance(features.index, pd.DatetimeIndex)

    basis = _load_aligned_feature(BASE_FEATURES["basis"], features.index)

    ctd_spread = None
    fut_spread = None
    if env_config.include_cost:
        ctd_spread = _load_aligned_feature(BASE_FEATURES["ctd_spread"], features.index)
        fut_spread = _load_aligned_feature(BASE_FEATURES["fut_spread"], features.index)

    return RLDataset(
        features=features,
        basis=basis,
        ctd_spread=ctd_spread,
        fut_spread=fut_spread,
    )


def _load_aligned_feature(base: BaseFeature, idx: pd.DatetimeIndex) -> pd.Series:
    s = _load_feature(base)
    s = s.loc[idx]

    if not s.index.equals(idx):
        raise ValueError(
            f"Loaded feature '{base.base_id}' could not be aligned to the dataset index"
        )

    return s
