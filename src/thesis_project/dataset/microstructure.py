from typing import Literal

import numpy as np
import pandas as pd

from thesis_project import config

# TODO: Refactor collectors/microstructure.py and collectors/relationships.py
# TODO: Leverage available code from utils and config
# TODO: Define a standard signature for single asset and cross asset functions
# TODO: Define utilities for common operations


def _validate_columns(lob: pd.DataFrame, columns: list[str]):
    missing = set(columns).difference(lob.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    return lob


def _validate_series(s) -> pd.Series:
    if not isinstance(s, pd.Series):
        raise TypeError(f"Expected a Series after squeeze, got {type(s)}")

    return s


def get_price(lob: pd.DataFrame, *, level: int, side: config.LobSide) -> pd.Series:
    columns = config.LOB.get_columns(levels=level, sides=side, column_types="price")
    lob = _validate_columns(lob, columns)

    s = lob[columns].squeeze("columns")
    s = _validate_series(s)
    s = s.rename(f"price_level_{level}_side_{side}")
    s = s.rename(index=config.LOB.index_name)

    return s


def get_size(lob: pd.DataFrame, *, level: int, side: config.LobSide) -> pd.Series:
    columns = config.LOB.get_columns(levels=level, sides=side, column_types="size")
    lob = _validate_columns(lob, columns)

    s = lob[columns].squeeze("columns")
    s = _validate_series(s)
    s = s.rename(f"size_level_{level}_side_{side}")
    s = s.rename(index=config.LOB.index_name)

    return s


def compute_mid_price(lob: pd.DataFrame) -> pd.Series:
    columns = config.LOB.get_columns(levels=[1], sides=("bid", "ask"), column_types="price")
    lob = _validate_columns(lob, columns)

    s = (lob["L1-AskPrice"] + lob["L1-BidPrice"]) / 2
    s = _validate_series(s)
    s = s.rename("mid_price")
    s = s.rename(index=config.LOB.index_name)

    return s


def compute_spread(lob: pd.DataFrame) -> pd.Series:
    columns = config.LOB.get_columns(levels=[1], sides=("bid", "ask"), column_types="price")
    lob = _validate_columns(lob, columns)

    s = (lob["L1-AskPrice"] - lob["L1-BidPrice"]) / 2
    s = _validate_series(s)
    s = s.rename("spread")
    s = s.rename(index=config.LOB.index_name)

    return s


def compute_micro_price(lob: pd.DataFrame) -> pd.Series:
    columns = config.LOB.get_columns(
        levels=[1], sides=("bid", "ask"), column_types=("price", "size")
    )
    lob = _validate_columns(lob, columns)

    num = lob["L1-AskPrice"] * lob["L1-BidSize"] + lob["L1-BidPrice"] * lob["L1-AskSize"]
    den = lob["L1-BidSize"] + lob["L1-AskSize"]

    s = num / den
    s = _validate_series(s)
    s = s.rename("micro_price")
    s = s.rename(index=config.LOB.index_name)

    return s


def compute_obi(lob: pd.DataFrame, *, max_level: int, ratio: bool) -> pd.Series:
    levels = list(range(1, max_level + 1, 1))
    ask_columns = config.LOB.get_columns(levels=levels, sides="ask", column_types="size")
    bid_columns = config.LOB.get_columns(levels=levels, sides="bid", column_types="size")
    lob = _validate_columns(lob, bid_columns + ask_columns)

    ask_depth = lob[ask_columns].fillna(0).sum(axis=1)
    bid_depth = lob[bid_columns].fillna(0).sum(axis=1)
    s = bid_depth - ask_depth
    if ratio:
        den = (bid_depth + ask_depth).replace(0, np.nan)
        s /= den

    s = _validate_series(s)

    suffix = "_ratio" if ratio else ""
    s = s.rename(f"obi_level_{max_level}{suffix}")
    s = s.rename(index=config.LOB.index_name)

    return s


def compute_bof(lob: pd.DataFrame, *, level: int) -> pd.Series:
    price_column = config.LOB.get_columns(levels=[level], sides="bid", column_types="price")
    size_column = config.LOB.get_columns(levels=[level], sides="bid", column_types="size")
    lob = _validate_columns(lob, price_column + size_column)

    price = lob[price_column]
    size = lob[size_column]
    s = np.select(
        [
            price > price.shift(1),
            price == price.shift(1),
            price < price.shift(1),
        ],
        [
            size + 0,
            size - size.shift(1),
            0 - size.shift(1),
        ],
        default=np.nan,
    )
    s = pd.Series(s, index=lob.index)
    s = _validate_series(s)
    s = s.rename(f"bof_level_{level}")
    s = s.rename(index=config.LOB.index_name)

    return s


def compute_aof(lob: pd.DataFrame, *, level: int) -> pd.Series:
    price_col = config.LOB.get_columns(levels=[level], sides="ask", column_types="price")
    size_col = config.LOB.get_columns(levels=[level], sides="ask", column_types="size")
    lob = _validate_columns(lob, price_col + size_col)

    price = lob[price_col]
    size = lob[size_col]
    s = np.select(
        [
            price > price.shift(1),
            price == price.shift(1),
            price < price.shift(1),
        ],
        [
            0 - size.shift(1),
            size - size.shift(1),
            size + 0,
        ],
        default=np.nan,
    )
    s = pd.Series(s, index=lob.index)
    s = _validate_series(s)
    s = s.rename(f"aof_level_{level}")
    s = s.rename(index=config.LOB.index_name)

    return s


def compute_ofi(lob: pd.DataFrame, *, level: int) -> pd.Series:
    bof = compute_bof(lob, level)
    aof = compute_aof(lob, level)

    s = bof + aof
    s = _validate_series(s)
    s = s.rename(f"ofi_level_{level}")
    s = s.rename(index=config.LOB.index_name)

    return s


def compute_slope_v1(lob: pd.DataFrame, *, max_level: int, side: config.LobSide) -> pd.Series:
    levels = list(range(1, max_level + 1, 1))
    price_columns = config.LOB.get_columns(levels=levels, sides=side, column_types="price")
    size_columns = config.LOB.get_columns(levels=levels, sides=side, column_types="size")
    _validate_columns(lob, price_columns + size_columns)

    n = len(levels)
    mid_price = compute_mid_price(lob)

    best_price_column = price_columns[0]
    best_size_column = size_columns[0]
    best_price = lob[best_price_column]
    best_size = lob[best_size_column]

    num1 = best_size
    den1 = np.abs((best_price / mid_price) - 1)
    s = num1 / den1

    if len(levels) > 1:
        num2 = np.abs((lob[size_columns[1:]] / lob[size_columns[:-1]]) - 1)
        den2 = np.abs((lob[price_columns[1:]] / lob[price_columns[:-1]]) - 1)
        s += (num2 / den2).sum(axis=1)

    s /= n
    s = _validate_series(s)
    s = s.rename(f"slope_v1_level_{max_level}")
    s = s.rename(index=config.LOB.index_name)

    return s


def compute_slope_v2(lob: pd.DataFrame, *, max_level: int, side: config.LobSide) -> pd.Series:
    levels = list(range(1, max_level + 1, 1))
    price_columns = config.LOB.get_columns(levels=levels, sides=side, column_types="price")
    size_columns = config.LOB.get_columns(levels=levels, sides=side, column_types="size")
    _validate_columns(lob, price_columns + size_columns)

    mid_price = compute_mid_price(lob)

    max_level_price_column = price_columns[-1]
    max_level_price = lob[max_level_price_column]

    num = lob[size_columns].fillna(0).sum(axis=1)
    den = np.abs(max_level_price - mid_price)

    s = num / den
    s = _validate_series(s)
    s = s.rename(f"slope_v2_level_{max_level}")
    s = s.rename(index=config.LOB.index_name)

    return s


SlopeType = Literal[
    "Naes and Skjeltorp (2006)",
    "Della Vedova, Gao, Grant and Westerholm (2021)",
]


def compute_slope(
    lob: pd.DataFrame, *, max_level: int, side: config.LobSide, slope_type: SlopeType
) -> pd.Series:
    match slope_type:
        case "Naes, Skjeltorp (2006)":
            return compute_slope_v1(lob, max_level=max_level, side=side)

        case "Della Vedova, Gao, Grant and Westerholm (2021)":
            return compute_slope_v2(lob, max_level=max_level, side=side)

        case _:
            raise ValueError("")
