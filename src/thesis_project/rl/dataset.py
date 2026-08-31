import datetime
from dataclasses import dataclass
from typing import get_args

import pandas as pd
from joblib import Parallel, delayed

from thesis_project import config, dataset, utils
from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.rl.env import RLDataset
from thesis_project.rl.env_config import EnvConfig


@dataclass(frozen=True)
class Settings:
    ticker: config.FutTicker
    n_jobs: int
    min_time: datetime.time = config.DEFAULT_OPENING_TIME
    max_time: datetime.time = config.DEFAULT_CLOSING_TIME

    offsets: dict[config.DateType, tuple[int, int] | None] = config.DEFAULT_EXCLUDED_DATES

    def __post_init__(self):
        if self.ticker not in get_args(config.FutTicker):
            raise ValueError("Ticker is not valid")

        if self.min_time >= self.max_time:
            raise ValueError("min_time must be earlier than max_time")

    def base_interval(self) -> tuple[datetime.time, datetime.time]:
        return (self.min_time, self.max_time)

    def preprocessing_interval(self, lookback: int) -> tuple[datetime.time, datetime.time]:
        min_time = utils.misc.sub_seconds_to_time(self.min_time, lookback)
        max_time = self.max_time

        return (min_time, max_time)


def build_spec(settings: Settings, spec: dataset.features.FeatureSpec) -> dict[str, pd.Series]:
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


def build_specs(settings: Settings, specs: list[dataset.features.FeatureSpec]) -> pd.DataFrame:
    dataset.features.validate_feature_specs(specs)
    features: dict[str, pd.Series] = {}
    results = Parallel(
        n_jobs=settings.n_jobs,
        backend="loky",
    )(delayed(build_spec)(spec, settings) for spec in specs)

    for result in results:
        features.update(result)  # type: ignore

    df = pd.DataFrame(features)
    min_time, max_time = settings.base_interval()
    df = df.between_time(min_time, max_time)
    utils.checks.check_df(df)

    return df


def build_rl_dataset(
    settings: Settings, env_config: EnvConfig, specs: list[dataset.features.FeatureSpec]
) -> RLDataset:
    features = build_specs(settings, specs)

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


def _load_aligned_feature(base: dataset.data.BaseFeature, idx: pd.DatetimeIndex) -> pd.Series:
    s = dataset.data.load_base_feature(base)
    s = s.loc[idx]

    if not s.index.equals(idx):
        raise ValueError(
            f"Loaded feature '{base.base_id}' could not be aligned to the dataset index"
        )

    return s
