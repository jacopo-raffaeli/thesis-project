import numpy as np
import pandas as pd

from thesis_project import config

# TODO: Refactor collectors/microstructure.py and collectors/relationships.py
# TODO: Leverage available code from utils and config
# TODO: Define a standard signature for single asset and cross asset functions
# TODO: Define utilities for common operations


def _validate_columns(lob: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns).difference(lob.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    return lob[columns]


def _validate_series(s) -> pd.Series:
    if not isinstance(s, pd.Series):
        raise TypeError(f"Expected a Series after squeeze, got {type(s)}")

    return s


def get_price(lob: pd.DataFrame, level: int, side: config.LobSide) -> pd.Series:
    columns = config.LOB.get_columns(levels=level, sides=side, column_types="price")
    lob = _validate_columns(lob, columns)

    s = lob[columns].squeeze("columns")
    s = _validate_series(s)
    s = s.rename(f"price_level_{level}_side_{side}")

    return s


def get_size(lob: pd.DataFrame, level: int, side: config.LobSide) -> pd.Series:
    columns = config.LOB.get_columns(levels=level, sides=side, column_types="size")
    lob = _validate_columns(lob, columns)

    s = lob[columns].squeeze("columns")
    s = _validate_series(s)
    s = s.rename(f"size_level_{level}_side_{side}")

    return s


def compute_mid_price(lob: pd.DataFrame) -> pd.Series:
    columns = config.LOB.get_columns(levels=[1], sides=("bid", "ask"), column_types="price")
    lob = _validate_columns(lob, columns)

    s = (lob["L1-AskPrice"] + lob["L1-BidPrice"]) / 2
    s = _validate_series(s)
    s = s.rename("mid_price")

    return s


def compute_spread(lob: pd.DataFrame) -> pd.Series:
    columns = config.LOB.get_columns(levels=[1], sides=("bid", "ask"), column_types="price")
    lob = _validate_columns(lob, columns)

    s = (lob["L1-AskPrice"] - lob["L1-BidPrice"]) / 2
    s = _validate_series(s)
    s = s.rename("spread")

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

    return s


def compute_obi(lob: pd.DataFrame, max_level: int, ratio: bool) -> pd.Series:
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
    s = s.rename(f"obi_level_{1}_{max_level}{suffix}")

    return s


def compute_bof(lob: pd.DataFrame) -> pd.Series: ...


def compute_aof(lob: pd.DataFrame) -> pd.Series: ...


def compute_ofi(lob: pd.DataFrame) -> pd.Series: ...


def compute_slope_v1(lob: pd.DataFrame) -> pd.Series: ...


def compute_slope_v2(lob: pd.DataFrame) -> pd.Series: ...
