import datetime
from dataclasses import dataclass
from typing import get_args

from thesis_project import config, utils


@dataclass(frozen=True)
class Settings:
    ticker: config.FutTicker
    min_time: datetime.time
    max_time: datetime.time
    n_jobs: int

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
