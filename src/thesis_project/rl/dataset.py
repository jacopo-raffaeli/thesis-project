from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from thesis_project import dataset, utils
from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.rl.dataset_config import DatasetConfig


@dataclass(frozen=True)
class RLDataset:
    # Features
    market_features: pd.DataFrame
    calendar_features: pd.DataFrame
    calendar_enc_features: pd.DataFrame

    # Market
    ctd_mid: pd.Series = field(default_factory=pd.Series)
    fut_mid: pd.Series = field(default_factory=pd.Series)
    ctd_contracts: float = field(default_factory=float)
    fut_contracts: pd.Series = field(default_factory=pd.Series)
    ctd_spread: pd.Series = field(default_factory=pd.Series)
    fut_spread: pd.Series = field(default_factory=pd.Series)

    # Temporal
    dates: list[pd.Timestamp] = field(default_factory=list[pd.Timestamp])
    date_to_slice: dict[pd.Timestamp, slice] = field(default_factory=dict[pd.Timestamp, slice])

    def __post_init__(self):
        # Check series consistency
        assert isinstance(self.market_features.index, pd.DatetimeIndex)
        assert isinstance(self.calendar_features.index, pd.DatetimeIndex)
        assert isinstance(self.calendar_enc_features.index, pd.DatetimeIndex)
        assert isinstance(self.ctd_mid.index, pd.DatetimeIndex)
        assert isinstance(self.fut_mid.index, pd.DatetimeIndex)
        assert isinstance(self.ctd_spread.index, pd.DatetimeIndex)
        assert isinstance(self.fut_spread.index, pd.DatetimeIndex)
        assert self.market_features.index.is_monotonic_increasing
        assert self.calendar_features.index.is_monotonic_increasing
        assert self.calendar_enc_features.index.is_monotonic_increasing
        assert self.ctd_mid.index.is_monotonic_increasing
        assert self.fut_mid.index.is_monotonic_increasing
        assert self.ctd_spread.index.is_monotonic_increasing
        assert self.fut_spread.index.is_monotonic_increasing
        assert self.market_features.index.equals(self.calendar_features.index)
        assert self.market_features.index.equals(self.calendar_enc_features.index)
        assert self.market_features.index.equals(self.ctd_mid.index)
        assert self.market_features.index.equals(self.fut_mid.index)
        assert self.market_features.index.equals(self.ctd_spread.index)
        assert self.market_features.index.equals(self.fut_spread.index)

        # Find unique dates
        dates_idx = self.market_features.index.tz_localize(None).floor("D")
        dates = dates_idx.unique()
        object.__setattr__(self, "dates", list(dates))

        # Map dates to indexes
        date_to_slice = {}
        for date in dates:
            rows = np.flatnonzero(dates_idx == date)
            date_to_slice[date] = slice(rows[0], rows[-1] + 1)
        object.__setattr__(self, "date_to_slice", date_to_slice)

    @property
    def n_market_features(self) -> int:
        return len(self.market_features.columns)

    @property
    def n_calendar_features(self) -> int:
        return len(self.calendar_features.columns)

    @property
    def n_calendar_enc_features(self) -> int:
        return len(self.calendar_enc_features.columns)

    @property
    def n_features(self) -> int:
        return self.n_market_features + self.n_calendar_features + self.n_calendar_enc_features

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def ctd_bid(self) -> pd.Series:
        return self.ctd_mid - (self.ctd_spread / 2)

    @property
    def ctd_ask(self) -> pd.Series:
        return self.ctd_mid + (self.ctd_spread / 2)

    @property
    def fut_bid(self) -> pd.Series:
        return self.fut_mid - (self.fut_spread / 2)

    @property
    def fut_ask(self) -> pd.Series:
        return self.fut_mid + (self.fut_spread / 2)


def build_spec(
    settings: DatasetConfig, spec: dataset.features_market.FeatureSpec
) -> dict[str, pd.Series]:
    # Load filtered base feature
    path = BASE_FEATURES[spec.base_id].path
    window = settings.preprocessing_interval(spec.lookback)
    excluded = utils.misc.get_dates_to_exclude(settings.ticker, settings.offsets)
    s = utils.io.load_filtered_parquet(path, time_window=window, dates_to_exclude=excluded)
    utils.checks.check_s(s)

    # Group by day
    assert isinstance(s.index, pd.DatetimeIndex)
    g = s.groupby(utils.misc.naive_dates(s.index))

    # Compute transforms
    transformed = {}
    for t in spec.transforms:
        transformed.update(t.transform(s, g, spec.base_id))

    return transformed


def build_specs(
    dataset_config: DatasetConfig,
) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    results = Parallel(
        n_jobs=dataset_config.n_jobs_market,
        backend="loky",
    )(delayed(build_spec)(dataset_config, spec) for spec in dataset_config.market_specs)

    for result in results:
        features.update(result)  # type: ignore

    df = pd.DataFrame(features)
    min_time, max_time = dataset_config.base_interval()
    df = df.between_time(min_time, max_time)
    utils.checks.check_df(df)

    return df


def build_rl_dataset(
    dataset_config: DatasetConfig,
) -> RLDataset:
    """ """
    # Market features
    market_features = build_specs(dataset_config).ffill(axis=0)
    index = market_features.index
    assert isinstance(index, pd.DatetimeIndex)

    # Calendar features
    calendar_features = dataset.features_temporal.compute_calendar_features(
        index, dataset_config.calendar_specs
    )

    # Calendar encoded features
    calendar_enc_features = dataset.features_temporal.compute_calendar_features_enc(
        index, dataset_config.calendar_enc_specs
    )

    # Mid prices
    ctd_mid = _load_aligned_feature(BASE_FEATURES["ctd_mid_price"], index).ffill(axis=0)
    fut_mid = _load_aligned_feature(BASE_FEATURES["fut_mid_price"], index).ffill(axis=0)

    # Contracts
    ctd_contracts = dataset_config.ctd_contracts
    cf = utils.io.load_cf(dataset_config.ticker)["CF"]
    fut_contracts = utils.misc.frac_fut_contracts(cf, ctd_contracts)
    if dataset_config.contract_mode == "round":
        fut_contracts = utils.misc.round_fut_contracts(fut_contracts)

    # Spreads
    ctd_spread = _load_aligned_feature(BASE_FEATURES["ctd_spread"], index).ffill(axis=0)
    fut_spread = _load_aligned_feature(BASE_FEATURES["fut_spread"], index).ffill(axis=0)

    # Build RL dataset
    return RLDataset(
        market_features=market_features,
        calendar_features=calendar_features,
        calendar_enc_features=calendar_enc_features,
        ctd_mid=ctd_mid,
        fut_mid=fut_mid,
        ctd_contracts=ctd_contracts,
        fut_contracts=fut_contracts,
        ctd_spread=ctd_spread,
        fut_spread=fut_spread,
    )


def split_rl_dataset(dataset: RLDataset, month: str) -> tuple[RLDataset, RLDataset]:
    # Split on the month
    split = pd.Period(month, "M")
    cutoff = split.end_time

    assert isinstance(dataset.market_features.index, pd.DatetimeIndex)
    index = dataset.market_features.index.tz_localize(None)

    train_mask = index <= cutoff
    test_mask = ~train_mask

    # Training set
    train = RLDataset(
        market_features=dataset.market_features[train_mask],
        calendar_features=dataset.calendar_features[train_mask],
        calendar_enc_features=dataset.calendar_enc_features[train_mask],
        ctd_mid=dataset.ctd_mid[train_mask],
        fut_mid=dataset.fut_mid[train_mask],
        ctd_contracts=dataset.ctd_contracts,
        fut_contracts=dataset.fut_contracts,
        ctd_spread=dataset.ctd_spread[train_mask],
        fut_spread=dataset.fut_spread[train_mask],
    )

    # Test set
    test = RLDataset(
        market_features=dataset.market_features[test_mask],
        calendar_features=dataset.calendar_features[test_mask],
        calendar_enc_features=dataset.calendar_enc_features[test_mask],
        ctd_mid=dataset.ctd_mid[test_mask],
        fut_mid=dataset.fut_mid[test_mask],
        ctd_contracts=dataset.ctd_contracts,
        fut_contracts=dataset.fut_contracts,
        ctd_spread=dataset.ctd_spread[test_mask],
        fut_spread=dataset.fut_spread[test_mask],
    )

    return train, test


def _load_aligned_feature(base: dataset.data.BaseFeature, idx: pd.DatetimeIndex) -> pd.Series:
    s = dataset.data.load_base_feature(base)
    s = s.loc[idx]

    if not s.index.equals(idx):
        raise ValueError(
            f"Loaded feature '{base.base_id}' could not be aligned to the dataset index"
        )

    return s


@dataclass(frozen=True)
class EpDataset:
    # Features
    market_features: pd.DataFrame
    calendar_features: pd.DataFrame
    calendar_enc_features: pd.DataFrame

    # Market
    ctd_mid: pd.Series = field(default_factory=pd.Series)
    fut_mid: pd.Series = field(default_factory=pd.Series)
    ctd_contracts: float = field(default_factory=float)
    fut_contracts: float = field(default_factory=float)
    ctd_spread: pd.Series = field(default_factory=pd.Series)
    fut_spread: pd.Series = field(default_factory=pd.Series)

    def __post_init__(self):
        # Check series consistency
        assert isinstance(self.market_features.index, pd.DatetimeIndex)
        assert isinstance(self.calendar_features.index, pd.DatetimeIndex)
        assert isinstance(self.calendar_enc_features.index, pd.DatetimeIndex)
        assert isinstance(self.ctd_mid.index, pd.DatetimeIndex)
        assert isinstance(self.fut_mid.index, pd.DatetimeIndex)
        assert isinstance(self.ctd_spread.index, pd.DatetimeIndex)
        assert isinstance(self.fut_spread.index, pd.DatetimeIndex)
        assert self.market_features.index.equals(self.calendar_features.index)
        assert self.market_features.index.equals(self.calendar_enc_features.index)
        assert self.market_features.index.equals(self.ctd_mid.index)
        assert self.market_features.index.equals(self.fut_mid.index)
        assert self.market_features.index.equals(self.ctd_spread.index)
        assert self.market_features.index.equals(self.fut_spread.index)

    @property
    def n_market_features(self) -> int:
        return len(self.market_features.columns)

    @property
    def n_calendar_features(self) -> int:
        return len(self.calendar_features.columns)

    @property
    def n_calendar_enc_features(self) -> int:
        return len(self.calendar_enc_features.columns)

    @property
    def n_features(self) -> int:
        return self.n_market_features + self.n_calendar_features + self.n_calendar_enc_features

    @property
    def episode_length(self) -> int:
        return len(self.market_features)

    @property
    def ctd_bid(self) -> pd.Series:
        return self.ctd_mid - (self.ctd_spread / 2)

    @property
    def ctd_ask(self) -> pd.Series:
        return self.ctd_mid + (self.ctd_spread / 2)

    @property
    def fut_bid(self) -> pd.Series:
        return self.fut_mid - (self.fut_spread / 2)

    @property
    def fut_ask(self) -> pd.Series:
        return self.fut_mid + (self.fut_spread / 2)
