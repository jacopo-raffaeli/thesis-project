import pandas as pd
from joblib import Parallel, delayed

from thesis_project import dataset, utils
from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.rl.dataset_config import DatasetConfig
from thesis_project.rl.env_trading import RLDataset

# TODO:
# - Improve the RLDataset splitting method


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
