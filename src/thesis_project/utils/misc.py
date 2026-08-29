import datetime as dt

import pandas as pd

from thesis_project import config


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
