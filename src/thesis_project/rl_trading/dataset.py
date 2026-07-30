from datetime import timedelta
from typing import Iterable

import pandas as pd

from thesis_project import config as global_config
from thesis_project.rl_trading.data import BASE_FEATURES
from thesis_project.rl_trading.dataset_config import DatasetConfig
from thesis_project.rl_trading.features import FEATURES, FeatureSpec


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


def _load_feature(feature: FeatureSpec) -> pd.Series:
    """
    Load a base feature and perform preliminary checks.
    """
    base = BASE_FEATURES[feature.base_id]
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

    return series


def _filter_time(idx: pd.DatetimeIndex, min_time: str, max_time: str) -> pd.DatetimeIndex:
    """
    Filter the index between min and max time. Time must be in hh:mm:ss format
    """
    mask = idx.indexer_between_time(min_time, max_time)

    return idx[mask]


def load_dates_to_exclude(ticker: str) -> dict[str, set[pd.Timestamp]]:
    """ """
    if ticker not in global_config.DATES_TO_EXCLUDE:
        raise ValueError(f"Ticker '{ticker}' not found")

    out = {}
    for key, dates in global_config.DATES_TO_EXCLUDE[ticker].items():
        out[key] = {pd.Timestamp(d).normalize() for d in dates}

    return out


def expand_dates_to_exclude(
    dates: Iterable[pd.Timestamp], before: int, after: int
) -> set[pd.Timestamp]:
    """ """
    out = set()
    for d in dates:
        out.update(d + timedelta(days=n) for n in range(-before, after + 1))

    return out


def get_dates_to_exclude(ticker: str, offsets: dict[str, tuple[int, int]]) -> set[pd.Timestamp]:
    """ """
    dates_dict = load_dates_to_exclude(ticker)

    # Check that the offsets given match an existing list of dates to exlcude
    for key in offsets:
        if key not in dates_dict:
            raise ValueError(f"Offset {key} not found")

    excluded = set()
    for key, dates in dates_dict.items():
        offset = offsets.get(key)

        if offset is None:
            excluded.update(dates)
        else:
            before, after = offset
            excluded.update(expand_dates_to_exclude(dates, before, after))

    return excluded


def _filter_dates(idx: pd.DatetimeIndex, excluded: set[pd.Timestamp]) -> pd.DatetimeIndex:
    """ """
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("idx must be a DatetimeIndex")

    dates = idx.normalize().tz_localize(None)
    mask = ~dates.isin(excluded)

    return idx[mask]


def _normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not idx.is_monotonic_increasing:
        idx = idx.sort_values()

    if idx.has_duplicates:
        idx = idx.drop_duplicates()

    if idx.hasnans:
        raise ValueError("NaNs found in the index")

    return idx


def build_dataset(config: DatasetConfig, specs: list[FeatureSpec]) -> pd.DataFrame:
    _validate_feature_specs(specs)

    excluded = get_dates_to_exclude(config.ticker, config.offsets)

    transformed = {}
    # TODO: Implement parallel version
    for spec in specs:
        s = _load_feature(spec)

        assert isinstance(s.index, pd.DatetimeIndex)
        idx = _normalize_index(s.index)
        idx = _filter_dates(idx, excluded)

        # TODO: Filter to properly compute transforms
        idx = _filter_time(idx, config.min_time, config.max_time)
        s = s[idx]

        if s.empty:
            raise ValueError("The Series is empty")

        g = s.groupby(pd.Grouper("D"))
        # TODO: Check seconds continuity, optionally fill with NaNs

        for transform in spec.transforms:
            transformed.update(transform.transform(s, g, spec.base_id))

    # TODO: Final filter
    df = pd.DataFrame().from_dict(transformed)

    if df.empty:
        raise ValueError("The DataFrame is empty")

    return df


if __name__ == "__main__":
    _validate_feature_specs(FEATURES)

    for feature in FEATURES:
        _load_feature(feature)
