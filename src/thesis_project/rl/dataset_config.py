import datetime
from dataclasses import dataclass, field
from typing import get_args

import pandas as pd

from thesis_project import config, utils
from thesis_project.utils.misc import get_dates_to_exclude


@dataclass(frozen=True)
class DatasetConfig:
    ticker: config.FutTicker

    min_time: datetime.time = datetime.time(9, 0, 0)
    max_time: datetime.time = datetime.time(17, 0, 0)

    min_time_data: datetime.time = config.BTP.market.opening_time
    max_time_data: datetime.time = config.BTP.market.closing_time

    offsets: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"fut_last_trading_days": (5, 0)}
    )
    dates_to_exclude: set[pd.Timestamp] = field(init=False)

    def __post_init__(self):
        if self.ticker not in get_args(config.FutTicker):
            raise ValueError("Ticker is not valid")

        if self.min_time >= self.max_time:
            raise ValueError("min_time must be earlier than max_time")

        for key in self.offsets:
            if key not in config.DATES_TO_EXCLUDE[self.ticker]:
                raise ValueError(f"Offset {key} not found")

        object.__setattr__(
            self, "dates_to_exclude", get_dates_to_exclude(self.ticker, self.offsets)
        )

    def get_base_interval(self) -> tuple[datetime.time, datetime.time]:
        return (self.min_time, self.max_time)

    def get_preprocessing_interval(self, lookback: int) -> tuple[datetime.time, datetime.time]:
        min_time = utils.misc.sub_seconds_to_time(self.min_time, lookback)
        max_time = self.max_time

        if min_time < self.min_time_data:
            raise ValueError("Lookback exceeds available data before market open")

        return (min_time, max_time)
