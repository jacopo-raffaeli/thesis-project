from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

from thesis_project import config as global_config


@dataclass(frozen=True)
class DatasetConfig:
    ticker: str

    min_time: str = "09:00:00"
    max_time: str = "17:00:00"

    min_time_data: str = str(global_config.CTD_MRKT_OPEN)
    max_time_data: str = str(global_config.CTD_MRKT_CLOSE)

    offsets: dict[str, tuple[int, int]] = field(default_factory=dict[str, tuple[int, int]])
    dates_to_exclude: set[pd.Timestamp] = field(init=False)

    def __post_init__(self):
        if self.ticker not in global_config.VALID_TICKERS:
            raise ValueError("Ticker is not valid")

        _validate_time(self.min_time)
        _validate_time(self.max_time)

        if pd.Timestamp(self.min_time) >= pd.Timestamp(self.max_time):
            raise ValueError("min_time must be earlier than max_time")

        for key in self.offsets:
            if key not in global_config.DATES_TO_EXCLUDE[self.ticker]:
                raise ValueError(f"Offset {key} not found")

        object.__setattr__(
            self, "dates_to_exclude", get_dates_to_exclude(self.ticker, self.offsets)
        )

    def get_base_interval(self) -> tuple[str, str]:
        return (self.min_time, self.max_time)

    def get_preprocessing_interval(self, lookback: int) -> tuple[str, str]:
        min_time = pd.Timestamp(self.min_time) - pd.Timedelta(seconds=lookback)
        max_time = self.max_time

        if min_time < pd.Timestamp(self.min_time_data):
            raise ValueError("Lookback exceeds available data before market open")

        min_time = min_time.strftime("%H:%M:%S")

        return (min_time, max_time)


def _validate_time(time: str) -> None:
    try:
        datetime.strptime(time, "%H:%M:%S")
    except ValueError as e:
        raise ValueError(f"Invalid time '{time}'. Expected HH:MM:SS.") from e


def load_dates_to_exclude(ticker: str) -> dict[str, set[pd.Timestamp]]:
    """ """
    out = {}
    for key, dates in global_config.DATES_TO_EXCLUDE[ticker].items():
        out[key] = {pd.Timestamp(d).normalize() for d in dates}

    return out


def expand_dates_to_exclude(
    dates: Iterable[pd.Timestamp], offset: tuple[int, int]
) -> set[pd.Timestamp]:
    """ """
    before, after = offset
    out = set()
    for d in dates:
        out.update(d + timedelta(days=n) for n in range(-before, after + 1))

    return out


def get_dates_to_exclude(ticker: str, offsets: dict[str, tuple[int, int]]) -> set[pd.Timestamp]:
    """ """
    dates_dict = load_dates_to_exclude(ticker)
    excluded = set()
    for key, dates in dates_dict.items():
        offset = offsets.get(key)

        if offset is None:
            excluded.update(dates)
        else:
            excluded.update(expand_dates_to_exclude(dates, offset))

    return excluded


if __name__ == "__main__":
    test = DatasetConfig(
        ticker="fbtp",
    )

    print(test.min_time_data)
    print(test.max_time_data)
