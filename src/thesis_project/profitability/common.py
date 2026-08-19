import datetime
from typing import get_args

import numpy as np
import pandas as pd
import polars as pl

from thesis_project import config
from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.profitability import settings
from thesis_project.profitability.settings import CTD_FACE_VALUE, FUT_FACE_VALUE
from thesis_project.rl_trading.dataset_config import DatasetConfig, get_dates_to_exclude


def load_prices(
    ticker: config.FutTicker, min_time: datetime.time, max_time: datetime.time
) -> pd.DataFrame:
    dataset_config = DatasetConfig(ticker)
    excluded = get_dates_to_exclude(dataset_config.ticker, dataset_config.offsets)
    excluded = [date.date() for date in excluded]

    names = {
        f"{role}_{side}_price_lvl_1"
        for role in get_args(config.AssetRole)
        for side in get_args(config.LobSide)
    }

    prices = {}
    for name in names:
        prices[name.replace("_lvl_1", "")] = (
            pl.scan_parquet(BASE_FEATURES[name].path)
            .filter(~pl.col("timestamp").dt.date().is_in(excluded))
            .filter(pl.col("timestamp").dt.time().is_between(min_time, max_time))
            .collect()
            .to_pandas()
            .set_index("timestamp")
            .squeeze("columns")
        )

    return pd.DataFrame(prices)


def load_volumes(
    ticker: config.FutTicker, min_time: datetime.time, max_time: datetime.time
) -> pd.DataFrame:
    dataset_config = DatasetConfig(ticker)
    excluded = get_dates_to_exclude(dataset_config.ticker, dataset_config.offsets)
    excluded = [date.date() for date in excluded]

    names = {
        f"{role}_{side}_size_lvl_1"
        for role in get_args(config.AssetRole)
        for side in get_args(config.LobSide)
    }

    volumes = {}
    for name in names:
        volumes[name.replace("_lvl_1", "")] = (
            pl.scan_parquet(BASE_FEATURES[name].path)
            .filter(~pl.col("timestamp").dt.date().is_in(excluded))
            .filter(pl.col("timestamp").dt.time().is_between(min_time, max_time))
            .collect()
            .to_pandas()
            .set_index("timestamp")
            .squeeze("columns")
        )

    return pd.DataFrame(volumes)


def load_market_data(
    ticker: config.FutTicker, min_time: datetime.time, max_time: datetime.time
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_prices(ticker, min_time, max_time)
    volumes = load_volumes(ticker, min_time, max_time)
    return prices, volumes


def naive_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.normalize()
    return index.tz_convert(None).normalize()


def count_days(index: pd.DatetimeIndex) -> int:
    return naive_dates(index).nunique()


def align_cf(prices: pd.DataFrame, daily_cf: pd.Series) -> pd.Series:
    assert isinstance(prices.index, pd.DatetimeIndex)

    cf = daily_cf.copy()
    cf.index = pd.to_datetime(cf.index).normalize()

    dates = prices.index.tz_convert(None).normalize()
    aligned = pd.Series(cf.reindex(dates).to_numpy(), index=prices.index, name="cf")

    if aligned.isna().any():
        missing_dates = dates[aligned.isna()].unique()
        raise ValueError(f"Missing conversion factor for dates: {missing_dates.tolist()}")

    return aligned


def compute_basis(
    ctd_price: pd.Series,
    fut_price: pd.Series,
    cf: pd.Series,
) -> pd.Series:
    if not ctd_price.index.equals(fut_price.index):
        raise ValueError("CTD and futures indexes do not match")
    if not ctd_price.index.equals(cf.index):
        raise ValueError("Price and conversion-factor indexes do not match")

    return ctd_price - cf * fut_price


def frac_fut_contracts(
    cf: pd.Series,
    ctd_contracts: int,
    ctd_face_value: float = CTD_FACE_VALUE,
    fut_face_value: float = FUT_FACE_VALUE,
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
    ctd_contracts: int,
    fut_contracts: pd.Series,
    ctd_face_value: float = CTD_FACE_VALUE,
    fut_face_value: float = FUT_FACE_VALUE,
) -> pd.Series:
    return (fut_contracts * fut_face_value / (ctd_contracts * ctd_face_value)).rename(
        "effective_cf"
    )


def execution_prices(
    data: pd.DataFrame,
    price: settings.PriceMode,
) -> dict[str, pd.Series]:
    match price:
        case "mid":
            return {
                "ctd_bid_price": (data["ctd_bid_price"] + data["ctd_ask_price"]) / 2,
                "ctd_ask_price": (data["ctd_bid_price"] + data["ctd_ask_price"]) / 2,
                "fut_bid_price": (data["fut_bid_price"] + data["fut_ask_price"]) / 2,
                "fut_ask_price": (data["fut_bid_price"] + data["fut_ask_price"]) / 2,
            }

        case "spread_enabled":
            return {
                "ctd_bid_price": data["ctd_bid_price"],
                "ctd_ask_price": data["ctd_ask_price"],
                "fut_bid_price": data["fut_bid_price"],
                "fut_ask_price": data["fut_ask_price"],
            }

        case _:
            raise ValueError(f"Unknown {price=}")


def execution_mask(
    data: pd.DataFrame,
    fut_contracts: pd.Series,
    ctd_contracts: int,
    *,
    liquidity: settings.LiquidityMode,
) -> pd.DataFrame:
    match liquidity:
        case "ignore":
            return pd.DataFrame(
                {
                    "long": True,
                    "short": True,
                },
                index=data.index,
            )

        case "level":
            return pd.DataFrame(
                {
                    "long": (
                        (data["ctd_ask_size"] >= ctd_contracts)
                        & (data["fut_bid_size"] >= fut_contracts)
                    ),
                    "short": (
                        (data["ctd_bid_size"] >= ctd_contracts)
                        & (data["fut_ask_size"] >= fut_contracts)
                    ),
                },
                index=data.index,
            )

        case "lob":
            raise NotImplementedError

        case _:
            raise ValueError(f"Unknown {liquidity=}")


# TODO: Move to utils
def shift_forward(series: pd.Series, horizon: int) -> pd.Series:
    target_index = series.index + pd.to_timedelta(horizon, unit="s")
    shifted = series.reindex(target_index)
    shifted.index = series.index
    return shifted


def has_event_between(
    entry_dates: pd.DatetimeIndex,
    exit_dates: pd.DatetimeIndex,
    event_dates: pd.DatetimeIndex,
    index: pd.DatetimeIndex,
    *,
    entry_inclusive: bool,
    exit_inclusive: bool,
) -> pd.Series:
    events = event_dates.sort_values().unique()

    if len(events) == 0:
        return pd.Series(False, index=index)

    left = np.searchsorted(
        events,
        entry_dates,
        side="left" if entry_inclusive else "right",
    )
    right = np.searchsorted(
        events,
        exit_dates,
        side="right" if exit_inclusive else "left",
    )

    return pd.Series(right > left, index=index)


def valid_trade_window(
    index: pd.DatetimeIndex,
    horizon: int,
    fut_last_trading_dates: pd.DatetimeIndex,
    ctd_switch_dates: pd.DatetimeIndex,
) -> pd.Series:
    entry_dates = naive_dates(index)
    exit_dates = naive_dates(index + pd.to_timedelta(horizon, unit="s"))

    invalid_fut = has_event_between(
        entry_dates,
        exit_dates,
        fut_last_trading_dates,
        index,
        entry_inclusive=True,
        exit_inclusive=False,
    )
    invalid_ctd = has_event_between(
        entry_dates,
        exit_dates,
        ctd_switch_dates,
        index,
        entry_inclusive=False,
        exit_inclusive=True,
    )

    return ~(invalid_fut | invalid_ctd)


def invalidate_overnight_trades(
    pnl: pd.DataFrame,
    horizon: int,
    fut_last_trading_dates: pd.DatetimeIndex,
    ctd_switch_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    assert isinstance(pnl.index, pd.DatetimeIndex)
    valid = valid_trade_window(
        pnl.index,
        horizon,
        fut_last_trading_dates,
        ctd_switch_dates,
    )
    return pnl.mask(~valid)
