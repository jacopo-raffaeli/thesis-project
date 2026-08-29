import logging
from typing import Callable, Literal, get_args

import numpy as np
import pandas as pd

from thesis_project import config, utils
from thesis_project.utils.checks import check_cols_in_df
from thesis_project.utils.misc import to_series

logger = logging.getLogger(__name__)

# TODO:
# - Add tests for feature formulas and edge cases.


def get_price(lob: pd.DataFrame, *, level: int, side: config.LobSide) -> pd.Series:
    """
    Extract the prices timeseries for a certain level and side for a daily LOB

    ## Args:
    * lob: DataFrame containing a preprocessed LOB
    * level: LOB level
    * side: LOB side

    ## Return:
    s: The extracted prices series
    """
    column = config.LOB.get_column(level=level, side=side, column_type="price")
    check_cols_in_df(lob, column)

    s = lob[column]
    s = to_series(s)

    return s


def get_size(lob: pd.DataFrame, *, level: int, side: config.LobSide) -> pd.Series:
    """
    Extract the sizes timeseries for a certain level and side for a daily LOB

    ## Args:
    * lob: DataFrame containing a preprocessed LOB
    * level: LOB level
    * side: LOB side

    ## Return:
    s: The extracted sizes series
    """
    column = config.LOB.get_column(level=level, side=side, column_type="size")
    check_cols_in_df(lob, column)

    s = lob[column]
    s = to_series(s)

    return s


def compute_mid_price(lob: pd.DataFrame) -> pd.Series:
    """
    Compute the mid prices timeseries for a daily LOB
    \nThe mid price is computed as: (L1-AskPrice + L1-BidPrice) / 2

    ## Args:
    * lob: DataFrame containing a preprocessed LOB

    ## Return:
    s: The mid price series
    """
    best_bid_price_column = config.LOB.get_column(level=1, side="bid", column_type="price")
    best_ask_price_column = config.LOB.get_column(level=1, side="ask", column_type="price")
    columns = [best_bid_price_column, best_ask_price_column]
    check_cols_in_df(lob, columns)

    best_bid_price = lob[best_bid_price_column]
    best_ask_price = lob[best_ask_price_column]

    s = (best_ask_price + best_bid_price) / 2
    s = to_series(s)

    return s


def compute_spread(lob: pd.DataFrame) -> pd.Series:
    """
    Compute the spreads timeseries for a daily LOB
    \nThe spread is computed as: L1-AskPrice - L1-BidPrice

    ## Args:
    * lob: DataFrame containing a preprocessed LOB

    ## Return:
    s: The spread series
    """
    best_bid_price_column = config.LOB.get_column(level=1, side="bid", column_type="price")
    best_ask_price_column = config.LOB.get_column(level=1, side="ask", column_type="price")
    columns = [best_bid_price_column, best_ask_price_column]
    check_cols_in_df(lob, columns)

    best_bid_price = lob[best_bid_price_column]
    best_ask_price = lob[best_ask_price_column]

    s = best_ask_price - best_bid_price
    s = to_series(s)

    return s


def compute_micro_price(lob: pd.DataFrame) -> pd.Series:
    """
    Compute the micro prices timeseries for a daily LOB
    \nThe micro price is computed as: (L1-AskPrice * L1-BidSize + L1-BidPrice * L1-AskSize) / (L1-Asksize + L1-BidSize)

    ## Args:
    * lob: DataFrame containing a preprocessed LOB

    ## Return:
    s: The micro price series
    """
    best_bid_price_column = config.LOB.get_column(level=1, side="bid", column_type="price")
    best_ask_price_column = config.LOB.get_column(level=1, side="ask", column_type="price")
    best_bid_size_column = config.LOB.get_column(level=1, side="bid", column_type="size")
    best_ask_size_column = config.LOB.get_column(level=1, side="ask", column_type="size")
    columns = [
        best_bid_price_column,
        best_ask_price_column,
        best_bid_size_column,
        best_ask_size_column,
    ]
    check_cols_in_df(lob, columns)

    best_bid_price = lob[best_bid_price_column]
    best_ask_price = lob[best_ask_price_column]
    best_bid_size = lob[best_bid_size_column]
    best_ask_size = lob[best_ask_size_column]

    num = best_ask_price * best_bid_size + best_bid_price * best_ask_size
    den = best_bid_size + best_ask_size

    s = num / den
    s = to_series(s)

    return s


def compute_obi(lob: pd.DataFrame, *, max_level: int, ratio: bool) -> pd.Series:
    """
    Compute the order book imbalance timeseries from 1 to max_level for a daily LOB
    \nThe OBI is computed as: sum_{i = 1}^{max_level} (L{i}-BidSize - L{i}-AskSize)
    \nIf ratio == True the OBI is normalized by sum_{i = 1}^{max_level} (L{i}-BidSize + L{i}-AskSize)

    ## Args:
    * lob: DataFrame containing a preprocessed LOB
    * max_level: The maximum level to include in the OBI
    * ratio: If True normalize the OBI

    ## Return:
    s: The OBI series
    """
    levels = list(range(1, max_level + 1, 1))
    ask_columns = config.LOB.get_columns(levels=levels, sides="ask", column_types="size")
    bid_columns = config.LOB.get_columns(levels=levels, sides="bid", column_types="size")
    columns = bid_columns + ask_columns
    check_cols_in_df(lob, columns)

    ask_depth = lob[ask_columns].fillna(0).sum(axis=1)
    bid_depth = lob[bid_columns].fillna(0).sum(axis=1)
    s = bid_depth - ask_depth
    if ratio:
        den = (bid_depth + ask_depth).replace(0, np.nan)
        s /= den

    s = to_series(s)

    return s


def compute_bof(lob: pd.DataFrame, *, level: int) -> pd.Series:
    """
    Compute the bid Order Flow timeseries for a certain level for a daily LOB
    \nThe bOF is computed as:
    \n

    ## Args:
    * lob: DataFrame containing a preprocessed LOB
    * level: LOB level

    ## Return:
    s: The bOF series
    """
    price_column = config.LOB.get_column(level=level, side="bid", column_type="price")
    size_column = config.LOB.get_column(level=level, side="bid", column_type="size")
    columns = [price_column, size_column]
    check_cols_in_df(lob, columns)

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
    s = to_series(s)

    return s


def compute_aof(lob: pd.DataFrame, *, level: int) -> pd.Series:
    """
    Compute the ask Order Flow timeseries for a certain level for a daily LOB
    \nThe aOF is computed as:
    \n

    ## Args:
    * lob: DataFrame containing a preprocessed LOB
    * level: LOB level

    ## Return:
    s: The aOF series
    """
    price_column = config.LOB.get_column(level=level, side="ask", column_type="price")
    size_column = config.LOB.get_column(level=level, side="ask", column_type="size")
    columns = [price_column, size_column]
    check_cols_in_df(lob, columns)

    price = lob[price_column]
    size = lob[size_column]
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
    s = to_series(s)

    return s


def compute_ofi(lob: pd.DataFrame, *, level: int) -> pd.Series:
    """
    Compute the Order Flow Imbalance timeseries for a certain level for a daily LOB
    \nThe OFI is computed as: bOF + aOF

    ## Args:
    * lob: DataFrame containing a preprocessed LOB
    * level: LOB level

    ## Return:
    s: The OFI series
    """
    bof = compute_bof(lob, level=level)
    aof = compute_aof(lob, level=level)

    s = bof + aof
    s = to_series(s)

    return s


def compute_slope_v1(lob: pd.DataFrame, *, max_level: int, side: config.LobSide) -> pd.Series:
    """
    Compute the Order Book slope timeseries from 1 to max_level for a certain side for a daily LOB
    \nThe slope is computed as descibed in: "Naes and Skjeltorp (2006)"

    ## Args:
    * lob: DataFrame containing a preprocessed LOB
    * max_level: The maximum level to include in the OBI
    * side: LOB side

    ## Return:
    s: The slope series
    """
    best_price_column = config.LOB.get_column(level=1, side=side, column_type="price")
    best_size_column = config.LOB.get_column(level=1, side=side, column_type="size")
    columns = [best_price_column, best_size_column]

    if max_level > 1:
        levels = list(range(1, max_level + 1, 1))
        price_columns = config.LOB.get_columns(levels=levels, sides=side, column_types="price")
        size_columns = config.LOB.get_columns(levels=levels, sides=side, column_types="size")
        columns = price_columns + size_columns

    check_cols_in_df(lob, columns)

    mid_price = compute_mid_price(lob)
    best_price = lob[best_price_column]
    best_size = lob[best_size_column]

    num1 = best_size
    den1 = (best_price / mid_price).sub(1).abs().replace(0, np.nan)
    s = num1 / den1

    if max_level > 1:
        num2 = (
            lob[size_columns]
            .div(lob[size_columns].shift(1, axis=1).replace(0, np.nan))
            .iloc[:, 1:]
            .sub(1)
        )
        den2 = (
            lob[price_columns]
            .div(lob[price_columns].shift(1, axis=1).replace(0, np.nan))
            .iloc[:, 1:]
            .sub(1)
            .abs()
            .replace(0, np.nan)
        )
        s += (num2 / den2).sum(axis=1)
        s /= max_level

    s = to_series(s)

    return s


def compute_slope_v2(lob: pd.DataFrame, *, max_level: int, side: config.LobSide) -> pd.Series:
    """
    Compute the Order Book slope timeseries from 1 to max_level for a certain side for a daily LOB
    \nThe slope is computed as descibed in: "Della Vedova, Gao, Grant and Westerholm (2021)"

    ## Args:
    * lob: DataFrame containing a preprocessed LOB
    * max_level: The maximum level to include in the OBI
    * side: LOB side

    ## Return:
    s: The slope series
    """
    levels = list(range(1, max_level + 1, 1))
    max_level_price_column = config.LOB.get_column(level=max_level, side=side, column_type="price")
    price_columns = config.LOB.get_columns(levels=levels, sides=side, column_types="price")
    size_columns = config.LOB.get_columns(levels=levels, sides=side, column_types="size")
    columns = price_columns + size_columns
    check_cols_in_df(lob, columns)

    mid_price = compute_mid_price(lob)
    max_level_price = lob[max_level_price_column]

    num = lob[size_columns].fillna(0).sum(axis=1)
    den = (max_level_price - mid_price).abs().replace(0, np.nan)

    s = num / den
    s = to_series(s)

    return s


SlopeType = Literal[
    "Naes and Skjeltorp (2006)",
    "Della Vedova, Gao, Grant and Westerholm (2021)",
]

SLOPE_DICT: dict[SlopeType, str] = {
    "Naes and Skjeltorp (2006)": "v1",
    "Della Vedova, Gao, Grant and Westerholm (2021)": "v2",
}


def compute_slope(
    lob: pd.DataFrame, *, max_level: int, side: config.LobSide, slope_type: SlopeType
) -> pd.Series:
    """
    Compute the Order Book slope timeseries from 1 to max_level for a certain side for a daily LOB
    \nThe slope is computed based on SlopeType passed

    ## Args:
    * lob: DataFrame containing a preprocessed LOB
    * max_level: The maximum level to include in the OBI
    * side: LOB side
    * slope_type: SlopeType literal

    ## Return:
    s: The slope series
    """
    match slope_type:
        case "Naes and Skjeltorp (2006)":
            return compute_slope_v1(lob, max_level=max_level, side=side)

        case "Della Vedova, Gao, Grant and Westerholm (2021)":
            return compute_slope_v2(lob, max_level=max_level, side=side)

        case _:
            raise ValueError("Unknown slope type")


def microstructure_feature_lobs(
    *,
    ticker: config.FutTicker,
    role: config.AssetRole,
    par_func: Callable[[pd.DataFrame], pd.Series],
) -> pd.Series:
    """
    Compute a microstructure feature over all LOBs of a certain ticker and role and concatentate them in a single series
    \nThe function must be partially initialized in all its arguments aside for the LOB DataFrame

    ## Args:
    * ticker: FutTicker literal
    * role: AssetRole literal
    * par_func: Partially initialized function with signature (DataFrame) -> Series

    ## Return:
    * s: Series of the microstructure feature
    """
    paths = utils.io.list_lob_paths(root=config.DATA_PRO_DIR, ticker=ticker, role=role)

    if not paths:
        base = config.DATA_PRO_DIR / ticker / role
        raise ValueError(f"No LOB found in {base!r}")

    s_daily: list[pd.Series] = []
    for path in paths:
        lob = pd.read_parquet(path)
        s_daily.append(par_func(lob))

    s = pd.concat(s_daily)

    return s


def microstructure_per_role(
    *,
    ticker: config.FutTicker,
    role: config.AssetRole,
    par_funcs: dict[tuple[str, str], Callable[[pd.DataFrame], pd.Series]],
):
    """
    Compute a set of microstructure feature over all LOBs of a certain ticker and role
    \nThe functions must be partially initialized in all its arguments aside for the LOB DataFrame
    \nThe Series are saved to data/processed/ticker/microstructure/folder/role_name.parquet

    ## Args:
    * ticker: FutTicker literal
    * role: AssetRole literal
    * par_funcs: Dict of keys (folder, name) and partially initialized functions
    """
    root = config.DATA_PRO_DIR / ticker / "microstructure"
    for (folder, name), par_func in par_funcs.items():
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{role}_{name}.parquet"

        s = microstructure_feature_lobs(ticker=ticker, role=role, par_func=par_func)
        utils.checks.check_series(
            s,
            # check_sampling=True
        )
        logger.debug("Computed: %s", filename.replace(".parquet", ""))

        s = s.to_frame(filename).rename_axis(config.LOB.index_name)
        # s.to_parquet(path / filename)
        logger.debug("Saved: %s", str((path / filename).relative_to(config.ROOT)))


def microstructure_per_ticker(
    *,
    ticker: config.FutTicker,
    par_funcs: dict[tuple[str, str], Callable[[pd.DataFrame], pd.Series]],
):
    """
    Compute a set of microstructure feature over all LOBs of a certain ticker
    \nThe functions must be partially initialized in all its arguments aside for the LOB DataFrame
    \nThe Series are saved to data/processed/ticker/microstructure/folder/role_name.parquet

    ## Args:
    * ticker: FutTicker literal
    * par_funcs: Dict of keys (folder, name) and partially initialized functions
    """
    for role in get_args(config.AssetRole):
        microstructure_per_role(ticker=ticker, role=role, par_funcs=par_funcs)


def cross_asset_per_ticker(
    *,
    ticker: config.FutTicker,
    par_funcs: dict[tuple[str, str], Callable[[config.FutTicker], pd.Series]],
):
    root = config.DATA_PRO_DIR / ticker / "microstructure"
    for (folder, name), par_func in par_funcs.items():
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{name}.parquet"

        s = par_func(ticker)
        utils.checks.check_series(
            s,
            # check_sampling=True
        )
        logger.debug("Computed: %s", filename.replace(".parquet", ""))

        s = s.to_frame(filename).rename_axis(config.LOB.index_name)
        # s.to_parquet(path / filename)
        logger.debug("Saved: %s", str((path / filename).relative_to(config.ROOT)))
