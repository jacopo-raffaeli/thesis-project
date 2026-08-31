import datetime
import datetime as dt

import pandas as pd

from thesis_project import config
from thesis_project.utils.io import load_criticalities, load_extra, load_fut_rollover_dates


def align_cf(prices: pd.DataFrame, daily_cf: pd.Series) -> pd.Series:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(prices.index).__name__!r}")

    if not isinstance(daily_cf.index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(daily_cf.index).__name__!r}")

    cf = daily_cf.copy()
    cf.index = pd.to_datetime(cf.index).normalize()

    dates = prices.index.tz_localize(None).normalize()
    aligned = pd.Series(cf.reindex(dates).to_numpy(), index=prices.index, name="cf")

    if cf.index.has_duplicates:
        raise ValueError("Conversion factor series contains duplicate dates")

    if aligned.isna().any():
        missing_dates = dates[aligned.isna()].unique()
        raise ValueError(f"Missing conversion factor for dates: {missing_dates.tolist()}")

    return aligned


def frac_fut_contracts(
    cf: pd.Series,
    ctd_contracts: float | int,
    ctd_face_value: float = config.BTP.contract_size,
    fut_face_value: float = config.FBTP.contract_size,
) -> pd.Series:
    return (cf * ctd_contracts * (ctd_face_value / fut_face_value)).rename("fut_contracts")


def round_fut_contracts(
    fut_contracts: pd.Series,
) -> pd.Series:
    fut_contracts = fut_contracts.round().astype(int)
    if (fut_contracts == 0).any():
        raise ValueError("Position size produces zero futures contracts")

    return fut_contracts.rename("fut_contracts")


def compute_eff_cf(
    ctd_contracts: float | int,
    fut_contracts: pd.Series | float | int,
    ctd_face_value: float = config.BTP.contract_size,
    fut_face_value: float = config.FBTP.contract_size,
) -> pd.Series | float:
    if ctd_contracts <= 0:
        raise ValueError("CTD contracts must be a positive number")

    if isinstance(fut_contracts, int | float) and fut_contracts <= 0:
        raise ValueError("FUT contracts must be a positive number")

    eff_cf = (fut_contracts * fut_face_value) / (ctd_contracts * ctd_face_value)

    if isinstance(eff_cf, pd.Series):
        eff_cf.rename("effective_cf")

    return eff_cf


def to_series(obj) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj

    if len(obj.columns) != 1:
        raise ValueError(f"Expected a single column DataFrame, got {len(obj.columns)} columns")

    obj = obj.squeeze("columns")
    assert isinstance(obj, pd.Series)

    return obj


def add_seconds_to_time(time: dt.time, seconds: int) -> dt.time:
    return (dt.datetime.combine(dt.date.today(), time) + dt.timedelta(seconds=seconds)).time()


def sub_seconds_to_time(time: dt.time, seconds: int) -> dt.time:
    return (dt.datetime.combine(dt.date.today(), time) - dt.timedelta(seconds=seconds)).time()


def _dates_by_type(ticker: config.FutTicker, name: config.DateType) -> list[datetime.date]:
    match name:
        case "Critical":
            return load_criticalities(ticker)

        case "Rollover":
            return load_fut_rollover_dates(ticker)

        case "Other":
            return load_extra(ticker)

        case _:
            raise ValueError("")


def expand_dates(dates: list[datetime.date], offset: tuple[int, int]) -> set[datetime.date]:
    """ """
    before, after = offset
    out = set()
    for d in dates:
        out.update(d + datetime.timedelta(days=n) for n in range(-before, after + 1))

    return out


def get_dates_to_exclude(
    ticker: config.FutTicker, dates_dict: dict[config.DateType, tuple[int, int] | None]
) -> list[datetime.date]:
    dates = set()
    for date_type, offset in dates_dict.items():
        if offset is None:
            dates.update(_dates_by_type(ticker, date_type))
        else:
            dates.update(expand_dates(_dates_by_type(ticker, date_type), offset))

    return sorted(dates)


def naive_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.normalize()
    return index.tz_localize(None).normalize()
