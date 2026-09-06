import datetime
from dataclasses import dataclass, field
from typing import get_args

from thesis_project import config, utils


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
