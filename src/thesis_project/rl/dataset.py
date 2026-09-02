import pandas as pd
from joblib import Parallel, delayed

from thesis_project import config, dataset, utils
from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.dataset.temporal import CalFeatEncType, CalFeatType
from thesis_project.rl.dataset_config import DatasetConfig
from thesis_project.rl.env import RLDataset
from thesis_project.rl.env_config import EnvConfig


def build_spec(settings: DatasetConfig, spec: dataset.features.FeatureSpec) -> dict[str, pd.Series]:
    # Load filtered base feature
    path = BASE_FEATURES[spec.base_id].path
    window = settings.preprocessing_interval(spec.lookback)
    excluded = utils.misc.get_dates_to_exclude(settings.ticker, config.DEFAULT_EXCLUDED_DATES)
    s = utils.io.load_filtered_parquet(path, window, excluded)
    utils.checks.check_s(s)

    # Group by day
    assert isinstance(s.index, pd.DatetimeIndex)
    g = s.groupby(utils.misc.naive_dates(s.index))

    # Compute transforms
    transformed = {}
    for t in spec.transforms:
        transformed.update(t.transform(s, g, spec.base_id))

    return transformed


def build_specs(settings: DatasetConfig, specs: list[dataset.features.FeatureSpec]) -> pd.DataFrame:
    dataset.features.validate_feature_specs(specs)
    features: dict[str, pd.Series] = {}
    results = Parallel(
        n_jobs=settings.n_jobs,
        backend="loky",
    )(delayed(build_spec)(settings, spec) for spec in specs)

    for result in results:
        features.update(result)  # type: ignore

    df = pd.DataFrame(features)
    min_time, max_time = settings.base_interval()
    df = df.between_time(min_time, max_time)
    utils.checks.check_df(df)

    return df


def build_rl_dataset(
    dataset_config: DatasetConfig,
    env_config: EnvConfig,
    specs: list[dataset.features.FeatureSpec],
    cal_feature: list[CalFeatType],
    cal_enc_feature: list[CalFeatEncType],
) -> RLDataset:
    # Market features
    market_features = build_specs(dataset_config, specs).ffill(axis=0)

    # Calendar features
    assert isinstance(market_features.index, pd.DatetimeIndex)
    calendar_features = dataset.temporal.compute_calendar_features(
        market_features.index, cal_feature
    )
    calendar_features_encoded = dataset.temporal.compute_calendar_features_encoded(
        market_features.index, cal_enc_feature
    )

    # Compose features dataset
    features = pd.concat(
        [market_features, calendar_features, calendar_features_encoded],
        axis=1,
    )

    # Mid prices
    assert isinstance(features.index, pd.DatetimeIndex)
    ctd_mid = _load_aligned_feature(BASE_FEATURES["ctd_mid_price"], features.index).ffill(axis=0)
    fut_mid = _load_aligned_feature(BASE_FEATURES["ctd_mid_price"], features.index).ffill(axis=0)

    # Contracts
    ctd_contracts = env_config.ctd_contracts
    cf = utils.io.load_cf(dataset_config.ticker)["CF"]
    fut_contracts = utils.misc.frac_fut_contracts(cf, ctd_contracts)
    if env_config.contract_mode == "round":
        fut_contracts = utils.misc.round_fut_contracts(fut_contracts)

    # Spreds
    ctd_spread = None
    fut_spread = None
    if env_config.price_mode == "quoted":
        ctd_spread = _load_aligned_feature(BASE_FEATURES["ctd_spread"], features.index).ffill(
            axis=0
        )
        fut_spread = _load_aligned_feature(BASE_FEATURES["fut_spread"], features.index).ffill(
            axis=0
        )

    return RLDataset(
        features=features,
        ctd_mid=ctd_mid,
        fut_mid=fut_mid,
        ctd_contracts=ctd_contracts,
        fut_contracts=fut_contracts,
        ctd_spread=ctd_spread,
        fut_spread=fut_spread,
    )


def split_rl_dataset(dataset: RLDataset, month: str) -> tuple[RLDataset, RLDataset]:
    split = pd.Period(month, "M")
    cutoff = split.end_time

    assert isinstance(dataset.features.index, pd.DatetimeIndex)
    index = dataset.features.index.tz_localize(None)

    train_mask = index <= cutoff
    test_mask = ~train_mask

    dataset1 = RLDataset(
        features=dataset.features[train_mask],
        ctd_mid=dataset.ctd_mid[train_mask],
        fut_mid=dataset.fut_mid[train_mask],
        ctd_contracts=dataset.ctd_contracts,
        fut_contracts=dataset.fut_contracts,
        ctd_spread=dataset.ctd_spread[train_mask] if dataset.ctd_spread is not None else None,
        fut_spread=dataset.fut_spread[train_mask] if dataset.fut_spread is not None else None,
    )

    dataset2 = RLDataset(
        features=dataset.features[test_mask],
        ctd_mid=dataset.ctd_mid[test_mask],
        fut_mid=dataset.fut_mid[test_mask],
        ctd_contracts=dataset.ctd_contracts,
        fut_contracts=dataset.fut_contracts,
        ctd_spread=dataset.ctd_spread[test_mask] if dataset.ctd_spread is not None else None,
        fut_spread=dataset.fut_spread[test_mask] if dataset.fut_spread is not None else None,
    )

    return dataset1, dataset2


def _load_aligned_feature(base: dataset.data.BaseFeature, idx: pd.DatetimeIndex) -> pd.Series:
    s = dataset.data.load_base_feature(base)
    s = s.loc[idx]

    if not s.index.equals(idx):
        raise ValueError(
            f"Loaded feature '{base.base_id}' could not be aligned to the dataset index"
        )

    return s
