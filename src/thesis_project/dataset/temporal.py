import datetime
from typing import Callable, Literal, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# Calendar features
def compute_year(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.year, index=idx, dtype=np.int16)


def compute_month(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.month, index=idx, dtype=np.int8)


def compute_week_of_year(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.isocalendar().week.astype(np.int32), index=idx)


def compute_week_of_month(idx: pd.DatetimeIndex) -> pd.Series:
    raise NotImplementedError("To be implemented")


def compute_day_of_year(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.dayofyear, index=idx, dtype=np.int16)


def compute_day_of_month(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.day, index=idx, dtype=np.int8)


def compute_day_of_week(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.dayofweek + 1, index=idx)


def compute_hour_of_day(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.hour, index=idx, dtype=np.int8)


def compute_minute_of_day(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.hour * 60 + idx.minute, index=idx, dtype=np.int16)


def compute_minute_of_hour(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.minute, index=idx, dtype=np.int8)


def compute_second_of_day(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.hour * 3600 + idx.minute * 60 + idx.second, index=idx, dtype=np.int32)


def compute_second_of_hour(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.minute * 60 + idx.second, index=idx, dtype=np.int16)


def compute_second_of_minute(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.second, index=idx, dtype=np.int8)


CalFeatType = Literal[
    "year",
    "month",
    "week_of_year",
    "day_of_year",
    "day_of_month",
    "day_of_week",
    "hour_of_day",
    "minute_of_day",
    "minute_of_hour",
    "second_of_day",
    "second_of_hour",
    "second_of_minute",
]


CALENDAR_FEATURES_REGISTRY: dict[CalFeatType, Callable[[pd.DatetimeIndex], pd.Series]] = {
    "year": compute_year,
    "month": compute_month,
    "week_of_year": compute_week_of_year,
    "day_of_year": compute_day_of_year,
    "day_of_month": compute_day_of_month,
    "day_of_week": compute_day_of_week,
    "hour_of_day": compute_hour_of_day,
    "minute_of_day": compute_minute_of_day,
    "minute_of_hour": compute_minute_of_hour,
    "second_of_day": compute_second_of_day,
    "second_of_hour": compute_second_of_hour,
    "second_of_minute": compute_second_of_minute,
}


# Calendar features encoded
def encode_cyclical(feature, period: int) -> Tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * (feature / period)
    return np.sin(angle), np.cos(angle)


def encode_hour_of_day(idx: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    feature = compute_hour_of_day(idx)
    sin, cos = encode_cyclical(feature, 24)
    return pd.Series(sin, index=idx), pd.Series(cos, index=idx)


def encode_minute_of_day(idx: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    feature = compute_minute_of_day(idx)
    sin, cos = encode_cyclical(feature, 24 * 60)
    return pd.Series(sin, index=idx), pd.Series(cos, index=idx)


def encode_minute_of_hour(idx: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    feature = compute_minute_of_hour(idx)
    sin, cos = encode_cyclical(feature, 60)
    return pd.Series(sin, index=idx), pd.Series(cos, index=idx)


def encode_second_of_day(idx: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    feature = compute_second_of_day(idx)
    sin, cos = encode_cyclical(feature, 24 * 60**2)
    return pd.Series(sin, index=idx), pd.Series(cos, index=idx)


def encode_second_of_hour(idx: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    feature = compute_second_of_hour(idx)
    sin, cos = encode_cyclical(feature, 60**2)
    return pd.Series(sin, index=idx), pd.Series(cos, index=idx)


def encode_second_of_minute(idx: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    feature = compute_second_of_minute(idx)
    sin, cos = encode_cyclical(feature, 60)
    return pd.Series(sin, index=idx), pd.Series(cos, index=idx)


CalFeatEncType = Literal[
    "hour_of_day",
    "minute_of_day",
    "minute_of_hour",
    "second_of_day",
    "second_of_hour",
    "second_of_minute",
]


CALENDAR_FEATURES_ENCODED_REGISTRY: dict[
    CalFeatEncType, Callable[[pd.DatetimeIndex], tuple[pd.Series, pd.Series]]
] = {
    "hour_of_day": encode_hour_of_day,
    "minute_of_day": encode_minute_of_day,
    "minute_of_hour": encode_minute_of_hour,
    "second_of_day": encode_second_of_day,
    "second_of_hour": encode_second_of_hour,
    "second_of_minute": encode_second_of_minute,
}


# Event based features
def sanitize_idx(idx: pd.DatetimeIndex):
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a pd.DatetimeIndex")

    if len(idx) == 0:
        raise ValueError("Index is empty")

    idx = idx.normalize().tz_localize(None)
    if not idx.is_monotonic_increasing:
        idx = idx.sort_values()

    return idx


def sanitize_dates(dates: pd.Series):
    if not isinstance(dates, pd.Series):
        raise TypeError("Dates must be a pd.Series")

    dates = pd.to_datetime(dates).dropna()
    if dates.dt.tz is not None:
        dates.dt.tz_localize(None)
    dates = dates.dt.normalize().drop_duplicates().sort_values()
    dates = dates.to_numpy()  # type: ignore

    if len(dates) == 0:
        raise ValueError("Dates is empty")

    return dates


def days_to_next_event(idx: pd.DatetimeIndex, dates: pd.Series):
    idx_out = idx
    idx = sanitize_idx(idx)
    dates = sanitize_dates(dates)

    if idx.max() > dates.max():
        raise ValueError("At least one date has not a matching next date")

    pos = dates.searchsorted(idx, side="left")
    dist = (dates[pos] - idx).days

    return pd.Series(dist, index=idx_out)


def days_to_prev_event(idx: pd.DatetimeIndex, dates: pd.Series, signed: bool = True):
    idx_out = idx
    idx = sanitize_idx(idx)
    dates = sanitize_dates(dates)

    if idx.min() < dates.min():
        raise ValueError("At least one date has not a matching previous date")

    pos = dates.searchsorted(idx, side="right") - 1
    dist = (idx - dates[pos]).days
    if signed:
        dist = -dist

    return pd.Series(dist, index=idx_out)


EventFeatType = Literal[
    "days_to_next",
    "days_to_prev",
]

EVENT_BASED_FEATURES_REGISTRY: dict[EventFeatType, Callable] = {
    "days_to_next": days_to_next_event,
    "days_to_prev": days_to_prev_event,
}


# Daily events
def time_to_daily_event(
    idx: pd.DatetimeIndex, event_time: str, event_tz: str, unit: str, signed: bool = True
) -> pd.Series:
    units = {"h": 3600, "m": 60, "s": 1}

    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a pd.DatetimeIndex")

    if idx.tz is None:
        raise TypeError("Index must be timezone aware")

    if not isinstance(event_time, str):
        raise TypeError("Event time must be a str (hh:mm:ss)")

    if unit not in units:
        raise ValueError(f"Invalid time unit: '{unit}', available time units are {units}")

    event_time_parsed = _parse_time(event_time)

    idx_event_tz = idx.tz_convert(ZoneInfo(event_tz))
    event_ts = idx_event_tz.normalize() + pd.Timedelta(
        hours=event_time_parsed.hour,
        minutes=event_time_parsed.minute,
        seconds=event_time_parsed.second,
    )

    time_to_event = (event_ts - idx_event_tz).total_seconds() / units[unit]
    time_to_event = time_to_event.astype(int)

    if not signed:
        time_to_event = np.abs(time_to_event)

    return pd.Series(time_to_event, index=idx)


def _parse_time(time_str: str) -> datetime.time:
    parts = [int(x) for x in time_str.split(":")]

    if len(parts) == 2:
        h, m = parts
        s = 0
    elif len(parts) == 3:
        h, m, s = parts
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    return datetime.time(hour=h, minute=m, second=s)


def compute_calendar_features(idx: pd.DatetimeIndex, ids: list[CalFeatType]) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for id in ids:
        func = CALENDAR_FEATURES_REGISTRY[id]
        features[id] = func(idx)

    return pd.DataFrame(features, index=idx)


def compute_calendar_features_encoded(
    idx: pd.DatetimeIndex, ids: list[CalFeatEncType]
) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for id in ids:
        func = CALENDAR_FEATURES_ENCODED_REGISTRY[id]
        sin_enc, cos_enc = func(idx)
        features[f"{id}_sin"] = sin_enc
        features[f"{id}_cos"] = cos_enc

    return pd.DataFrame(features, index=idx)
