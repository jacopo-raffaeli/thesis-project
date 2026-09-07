import datetime
from dataclasses import dataclass, field
from typing import get_args

from thesis_project import config, utils
from thesis_project.dataset.features_market import FeatureSpec, validate_feature_specs
from thesis_project.dataset.features_temporal import (
    CalFeatEncType,
    CalFeatType,
    validate_calendar_encoded_features,
    validate_calendar_features,
)


@dataclass(frozen=True)
class DatasetConfig:
    ticker: config.FutTicker
    n_jobs: int
    contract_mode: config.ContractMode

    min_time: datetime.time = config.DEFAULT_OPENING_TIME
    max_time: datetime.time = config.DEFAULT_CLOSING_TIME
    offsets: dict[config.DateType, tuple[int, int] | None] = field(
        default_factory=lambda: config.DEFAULT_EXCLUDED_DATES
    )

    ctd_contracts: float = 1.0

    market: list[FeatureSpec] = field(default_factory=list)
    calendar: list[CalFeatType] = field(default_factory=list)
    calendar_enc: list[CalFeatEncType] = field(default_factory=list)

    def __post_init__(self):
        if self.ticker not in get_args(config.FutTicker):
            raise ValueError("Ticker is not valid")

        if self.min_time >= self.max_time:
            raise ValueError("min_time must be earlier than max_time")

        if self.market:
            validate_feature_specs(self.market)

        if self.calendar:
            validate_calendar_features(self.calendar)

        if self.calendar_enc:
            validate_calendar_encoded_features(self.calendar_enc)

    def base_interval(self) -> tuple[datetime.time, datetime.time]:
        return (self.min_time, self.max_time)

    def preprocessing_interval(self, lookback: int) -> tuple[datetime.time, datetime.time]:
        min_time = utils.misc.sub_seconds_to_time(self.min_time, lookback)
        max_time = self.max_time

        return (min_time, max_time)

    @property
    def n_market_features(self) -> int:
        count = 0
        for spec in self.market:
            count += spec.n_outputs

        return count

    @property
    def n_calendar_features(self) -> int:
        return len(self.calendar)

    @property
    def n_calendar_enc_features(self) -> int:
        return 2 * len(self.calendar)

    @property
    def n_features(self) -> int:
        return self.n_market_features + self.n_calendar_features + self.n_calendar_enc_features

    @property
    def features_names(self) -> list[str]:
        names = []
        for spec in self.market:
            names.extend(spec.output_names)

        for name in self.calendar:
            names.append(name)

        for name in self.calendar_enc:
            names.append(f"{name}_sin")
            names.append(f"{name}_cos")

        return names
