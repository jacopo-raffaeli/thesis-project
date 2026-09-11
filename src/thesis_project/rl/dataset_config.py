import datetime
from dataclasses import dataclass, field
from typing import get_args

from thesis_project import config, utils
from thesis_project.dataset.features_market import FeatureSpec, validate_feature_specs
from thesis_project.dataset.features_temporal import (
    CalEncFeatType,
    CalFeatType,
    validate_calendar_encoded_features,
    validate_calendar_features,
)
from thesis_project.rl import features


@dataclass(frozen=True)
class DatasetConfig:
    ticker: config.FutTicker
    contract_mode: config.ContractMode
    n_jobs_market: int | None = None

    min_time: datetime.time = config.DEFAULT_OPENING_TIME
    max_time: datetime.time = config.DEFAULT_CLOSING_TIME
    offsets: dict[config.DateType, tuple[int, int] | None] = field(
        default_factory=lambda: config.DEFAULT_EXCLUDED_DATES.copy()
    )

    ctd_contracts: float = 1.0

    market_set: str = "xgb_cls"
    calendar_set: str = "default"
    calendar_enc_set: str = "default"

    def __post_init__(self):
        if self.ticker not in get_args(config.FutTicker):
            raise ValueError(self.ticker)

        if self.ctd_contracts <= 0:
            raise ValueError("ctd_contracts must be > 0")

        if self.min_time >= self.max_time:
            raise ValueError("min_time must be earlier than max_time")

        if self.market_specs:
            validate_feature_specs(self.market_specs)

        if self.calendar_specs:
            validate_calendar_features(self.calendar_specs)

        if self.calendar_enc_specs:
            validate_calendar_encoded_features(self.calendar_enc_specs)

        if self.n_jobs_market is None:
            object.__setattr__(
                self,
                "n_jobs_market",
                len(self.market_specs),
            )

    def base_interval(self) -> tuple[datetime.time, datetime.time]:
        return (self.min_time, self.max_time)

    def preprocessing_interval(self, lookback: int) -> tuple[datetime.time, datetime.time]:
        min_time = utils.misc.sub_seconds_to_time(self.min_time, lookback)
        max_time = self.max_time

        return (min_time, max_time)

    @property
    def market_specs(self) -> list[FeatureSpec]:
        return features.get_market_feature_set(self.market_set)

    @property
    def calendar_specs(self) -> list[CalFeatType]:
        return features.get_calendar_feature_set(self.calendar_set)

    @property
    def calendar_enc_specs(self) -> list[CalEncFeatType]:
        return features.get_calendar_enc_feature_set(self.calendar_enc_set)

    @property
    def n_market_features(self) -> int:
        count = 0
        for spec in self.market_specs:
            count += spec.n_outputs

        return count

    @property
    def n_calendar_features(self) -> int:
        return len(self.calendar_specs)

    @property
    def n_calendar_enc_features(self) -> int:
        return 2 * len(self.calendar_enc_specs)

    @property
    def n_features(self) -> int:
        return self.n_market_features + self.n_calendar_features + self.n_calendar_enc_features

    @property
    def features_names(self) -> list[str]:
        names = []
        for spec in self.market_specs:
            names.extend(spec.output_names)

        for name in self.calendar_specs:
            names.append(name)

        for name in self.calendar_enc_specs:
            names.append(f"{name}_sin")
            names.append(f"{name}_cos")

        return names
