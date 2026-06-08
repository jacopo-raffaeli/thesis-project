import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from thesis_project import config


def compute_bid_ask_price(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute (i.e. extract) bid/ask price for a specific level of the LOB.

    Args:
        lob_df: DataFrame with LOB data
        filename: Name of the file being processed
        lob_type: Type of the LOB ("futures" or "ctd")
        context: dict with requried keys:
            - 'LOB level': level of the LOB to extract bid/ask from
            - 'Side': "bid" or "ask"

    Returns:
        DataFrame with bid/ask price series
    """
    context = context or {}

    if "LOB level" not in context:
        warnings.warn(f"{filename}: Context 'LOB level' is missing", UserWarning)
        return None

    if "Side" not in context:
        warnings.warn(f"{filename}: Context 'Side' is missing", UserWarning)
        return None

    lvl = context.get("LOB level")
    if lvl not in range(1, config.LEVELS + 1):
        warnings.warn(
            f"{filename}: Context 'LOB level' ({lvl}) out of range 1 - {config.LEVELS}", UserWarning
        )
        return None

    side = context.get("Side")
    if side not in {"bid", "ask"}:
        warnings.warn(
            f"{filename}: Context 'Side' ({side}) is invalid. Expected 'bid' or 'ask'", UserWarning
        )
        return None

    if side == "bid":
        col = config.BID_PRICE_COL_TEMPLATE.format(lvl=lvl)
    else:
        col = config.ASK_PRICE_COL_TEMPLATE.format(lvl=lvl)

    required_cols = {
        col,
    }

    if not required_cols.issubset(lob_df.columns):
        warnings.warn(
            f"{filename}: The following required columns are missing: {', '.join(required_cols - set(lob_df.columns))}",
            UserWarning,
        )
        return None

    price = lob_df[col]

    out_df = pd.DataFrame(
        {
            "Price": price,
        },
        index=lob_df.index,
    )
    out_df.index.name = lob_df.index.name

    return out_df


def compute_bid_ask_size(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute (i.e. extract) bid/ask size for a specific level of the LOB.

    Args:
        lob_df: DataFrame with LOB data
        filename: Name of the file being processed
        lob_type: Type of the LOB ("futures" or "ctd")
        context: dict with requried keys:
            - 'LOB level': level of the LOB to extract bid/ask from
            - 'Side': "bid" or "ask"

    Returns:
        DataFrame with bid/ask size series
    """
    context = context or {}

    if "LOB level" not in context:
        warnings.warn(f"{filename}: Context 'LOB level' is missing", UserWarning)
        return None

    if "Side" not in context:
        warnings.warn(f"{filename}: Context 'Side' is missing", UserWarning)
        return None

    lvl = context.get("LOB level")
    if lvl not in range(1, config.LEVELS + 1):
        warnings.warn(
            f"{filename}: Context 'LOB level' ({lvl}) out of range 1 - {config.LEVELS}", UserWarning
        )
        return None

    side = context.get("Side")
    if side not in {"bid", "ask"}:
        warnings.warn(
            f"{filename}: Context 'Side' ({side}) is invalid. Expected 'bid' or 'ask'", UserWarning
        )
        return None

    if side == "bid":
        col = config.BID_SIZE_COL_TEMPLATE.format(lvl=lvl)
    else:
        col = config.ASK_SIZE_COL_TEMPLATE.format(lvl=lvl)

    required_cols = {
        col,
    }

    if not required_cols.issubset(lob_df.columns):
        warnings.warn(
            f"{filename}: The following required columns are missing: {', '.join(required_cols - set(lob_df.columns))}",
            UserWarning,
        )
        return None

    size = lob_df[col]

    out_df = pd.DataFrame(
        {
            "Size": size,
        },
        index=lob_df.index,
    )
    out_df.index.name = lob_df.index.name

    return out_df


def compute_spread(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute spread between bid and ask for a specific level of the LOB.

    Args:
        lob_df: DataFrame with LOB data
        filename: Name of the file being processed
        lob_type: Type of the LOB ("futures" or "ctd")
        context: dict containing 'LOB level' with the level to compute the spread for

    Returns:
        DataFrame with spread series
    """
    context = context or {}

    if "LOB level" not in context:
        warnings.warn(f"{filename}: Context 'LOB level' is missing", UserWarning)
        return None

    lvl = context.get("LOB level")
    if lvl not in range(1, config.LEVELS + 1):
        warnings.warn(
            f"{filename}: Context 'LOB level' ({lvl}) out of range 1 - {config.LEVELS}", UserWarning
        )
        return None

    bid_col = config.BID_PRICE_COL_TEMPLATE.format(lvl=lvl)
    ask_col = config.ASK_PRICE_COL_TEMPLATE.format(lvl=lvl)
    required_cols = {
        bid_col,
        ask_col,
    }

    if not required_cols.issubset(lob_df.columns):
        warnings.warn(
            f"{filename}: The following required columns are missing: {', '.join(required_cols - set(lob_df.columns))}",
            UserWarning,
        )
        return None

    spread = lob_df[ask_col] - lob_df[bid_col]
    if spread.empty:
        warnings.warn(f"{filename}: Spread series is empty", UserWarning)
        return None

    out_df = pd.DataFrame(
        {
            "Spread": spread,
        },
        index=lob_df.index,
    )
    out_df.index.name = lob_df.index.name

    return out_df


def compute_mid_price(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute mid price between bid and ask for a specific level of the LOB.

    Args:
        lob_df: DataFrame with LOB data
        filename: Name of the file being processed
        lob_type: Type of the LOB ("futures" or "ctd")
        context: dict containing 'LOB level' with the level to compute the spread for

    Returns:
        DataFrame with spread series
    """
    context = context or {}

    if "LOB level" not in context:
        warnings.warn(f"{filename}: Context 'LOB level' is missing", UserWarning)
        return None

    lvl = context.get("LOB level")
    if lvl not in range(1, config.LEVELS + 1):
        warnings.warn(
            f"{filename}: Context 'LOB level' ({lvl}) out of range 1 - {config.LEVELS}", UserWarning
        )
        return None

    bid_col = config.BID_PRICE_COL_TEMPLATE.format(lvl=lvl)
    ask_col = config.ASK_PRICE_COL_TEMPLATE.format(lvl=lvl)
    required_cols = {
        bid_col,
        ask_col,
    }

    if not required_cols.issubset(lob_df.columns):
        warnings.warn(
            f"{filename}: The following required columns are missing: {', '.join(required_cols - set(lob_df.columns))}",
            UserWarning,
        )
        return None

    mid = (lob_df[ask_col] + lob_df[bid_col]) / 2
    if mid.empty:
        warnings.warn(f"{filename}: Mid price series is empty", UserWarning)
        return None

    out_df = pd.DataFrame(
        {
            "Mid Price": mid,
        },
        index=lob_df.index,
    )
    out_df.index.name = lob_df.index.name

    return out_df


def compute_micro_price(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute micro price between bid and ask for a specific level of the LOB.

    Args:
        lob_df: DataFrame with LOB data
        filename: Name of the file being processed
        lob_type: Type of the LOB ("futures" or "ctd")
        context: dict containing keys:
            - 'LOB level' with the level to compute the spread for

    Returns:
        DataFrame with spread series
    """
    context = context or {}

    if "LOB level" not in context:
        warnings.warn(f"{filename}: Context 'LOB level' is missing", UserWarning)
        return None

    lvl = context.get("LOB level")
    if lvl not in range(1, config.LEVELS + 1):
        warnings.warn(
            f"{filename}: Context 'LOB level' ({lvl}) out of range 1 - {config.LEVELS}", UserWarning
        )
        return None

    bid_col = config.BID_PRICE_COL_TEMPLATE.format(lvl=lvl)
    ask_col = config.ASK_PRICE_COL_TEMPLATE.format(lvl=lvl)
    bid_vol_col = config.BID_SIZE_COL_TEMPLATE.format(lvl=lvl)
    ask_vol_col = config.ASK_SIZE_COL_TEMPLATE.format(lvl=lvl)
    required_cols = {
        bid_col,
        ask_col,
        bid_vol_col,
        ask_vol_col,
    }

    if not required_cols.issubset(lob_df.columns):
        warnings.warn(
            f"{filename}: The following required columns are missing: {', '.join(required_cols - set(lob_df.columns))}",
            UserWarning,
        )
        return None

    num = lob_df[ask_col] * lob_df[bid_vol_col] + lob_df[bid_col] * lob_df[ask_vol_col]
    den = lob_df[bid_vol_col] + lob_df[ask_vol_col]

    if (den == 0).any():
        warnings.warn(
            f"{filename}: Denominator for micro price contains zeros. Replaced with NaNs",
            UserWarning,
        )
        den.replace(0, np.nan, inplace=True)

    micro = num / den

    if micro.empty:
        warnings.warn(f"{filename}: Micro price series is empty", UserWarning)
        return None

    if micro.isna().all():
        warnings.warn(f"{filename}: Micro price series is all NaN", UserWarning)
        return None

    if (micro == np.inf).any() or (micro == -np.inf).any():
        warnings.warn(
            f"{filename}: Micro price series contains infinite values. Replaced with NaNs",
            UserWarning,
        )
        micro.replace([np.inf, -np.inf], np.nan, inplace=True)

    if micro.isna().all():
        warnings.warn(
            f"{filename}: Micro price series is all NaN after replacing inf with NaN",
            UserWarning,
        )
        return None

    out_df = pd.DataFrame(
        {
            "Micro Price": micro,
        },
        index=lob_df.index,
    )
    out_df.index.name = lob_df.index.name

    return out_df


def compute_obi(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute Order Book Imbalance (OBI) for a specific range of LOB levels.

    Args:
        lob_df: DataFrame with LOB data
        filename: Name of the file being processed
        lob_type: Type of the LOB ("futures" or "ctd")
        context: dict containing keys:
            - 'LOB level range' with the levels to compute OBI for
            - 'Scale' boolean indicating whether to scale OBI by total volume

    Returns:
        DataFrame with OBI series
    """
    context = context or {}

    if "LOB level range" not in context:
        warnings.warn(f"{filename}: Context 'LOB level range' is missing", UserWarning)
        return None

    levels = context.get("LOB level range", 1)
    if not all(lvl in range(1, config.LEVELS + 1) for lvl in levels):
        warnings.warn(
            f"{filename}: Context 'LOB level range' contains invalid levels. Expected levels in range 1 - {config.LEVELS}",
            UserWarning,
        )
        return None

    if "Scale" not in context:
        warnings.warn(f"{filename}: Context 'Scale' is missing", UserWarning)
        return None

    scale = context.get("Scale", True)
    if not isinstance(scale, bool):
        warnings.warn(f"{filename}: Context 'Scale' should be a boolean value", UserWarning)
        return None

    required_cols = set()
    for lvl in levels:
        bid_vol_col = config.BID_SIZE_COL_TEMPLATE.format(lvl=lvl)
        ask_vol_col = config.ASK_SIZE_COL_TEMPLATE.format(lvl=lvl)

        required_cols.update({bid_vol_col, ask_vol_col})

    if not required_cols.issubset(lob_df.columns):
        warnings.warn(
            f"{filename}: The following required columns are missing: {', '.join(required_cols - set(lob_df.columns))}",
            UserWarning,
        )
        return None

    bid_cols = [config.BID_SIZE_COL_TEMPLATE.format(lvl=lvl) for lvl in levels]
    ask_cols = [config.ASK_SIZE_COL_TEMPLATE.format(lvl=lvl) for lvl in levels]
    bid_vol = lob_df[bid_cols].sum(axis=1)
    ask_vol = lob_df[ask_cols].sum(axis=1)
    obi = bid_vol - ask_vol

    if scale:
        den = bid_vol + ask_vol
        if (den == 0).any():
            # bid_zeros = (bid_vol == 0).mean()
            # ask_zeros = (ask_vol == 0).mean()
            # warnings.warn(
            #     (f"{filename}: Denominator for OBI scaling contains zeros. Replaced with NaNs."
            #      f"\nProportion of zero bid volume: {bid_zeros:.2%}, zero ask volume: {ask_zeros:.2%}"),
            #     UserWarning,
            # )
            den.replace(0, np.nan, inplace=True)
        obi /= den

    if obi.empty:
        warnings.warn(f"{filename}: OBI series is empty", UserWarning)
        return None

    if obi.isna().all():
        warnings.warn(f"{filename}: OBI series is all NaN", UserWarning)
        return None

    if (obi == np.inf).any() or (obi == -np.inf).any():
        warnings.warn(
            f"{filename}: OBI series contains infinite values. Replaced with NaNs",
            UserWarning,
        )
        obi.replace([np.inf, -np.inf], np.nan, inplace=True)

    if obi.isna().all():
        warnings.warn(
            f"{filename}: OBI series is all NaN after replacing inf with NaN",
            UserWarning,
        )
        return None

    out_df = pd.DataFrame(
        {
            "OBI": obi,
        },
        index=lob_df.index,
    )
    out_df.index.name = lob_df.index.name

    return out_df


def compute_ofi(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    context = context or {}

    if "LOB level" not in context:
        warnings.warn(f"{filename}: Context 'LOB level' is missing", UserWarning)
        return None

    lvl = context.get("LOB level")
    if lvl not in range(1, config.LEVELS + 1):
        warnings.warn(
            f"{filename}: Context 'LOB level' ({lvl}) out of range 1 - {config.LEVELS}", UserWarning
        )
        return None

    bid_price_col = config.BID_PRICE_COL_TEMPLATE.format(lvl=lvl)
    ask_price_col = config.ASK_PRICE_COL_TEMPLATE.format(lvl=lvl)
    bid_vol_col = config.BID_SIZE_COL_TEMPLATE.format(lvl=lvl)
    ask_vol_col = config.ASK_SIZE_COL_TEMPLATE.format(lvl=lvl)
    required_cols = {
        bid_price_col,
        ask_price_col,
        bid_vol_col,
        ask_vol_col,
    }
    if not required_cols.issubset(lob_df.columns):
        warnings.warn(
            f"{filename}: The following required columns are missing: {', '.join(required_cols - set(lob_df.columns))}",
            UserWarning,
        )
        return None

    p_b_prev = lob_df[bid_price_col].shift(1)
    p_a_prev = lob_df[ask_price_col].shift(1)
    v_b_prev = lob_df[bid_vol_col].shift(1)
    v_a_prev = lob_df[ask_vol_col].shift(1)

    out_df = pd.DataFrame(index=lob_df.index)

    out_df["bOF"] = np.select(
        [
            lob_df[bid_price_col] > p_b_prev,
            lob_df[bid_price_col] == p_b_prev,
            lob_df[bid_price_col] < p_b_prev,
        ],
        [
            lob_df[bid_vol_col],
            lob_df[bid_vol_col] - v_b_prev,
            -v_b_prev,
        ],
        default=np.nan,
    )

    out_df["aOF"] = np.select(
        [
            lob_df[ask_price_col] > p_a_prev,
            lob_df[ask_price_col] == p_a_prev,
            lob_df[ask_price_col] < p_a_prev,
        ],
        [
            -v_a_prev,
            lob_df[ask_vol_col] - v_a_prev,
            lob_df[ask_vol_col],
        ],
        default=np.nan,
    )

    out_df["OFI"] = out_df["bOF"] + out_df["aOF"]

    if out_df["OFI"].empty:
        warnings.warn(f"{filename}: OFI series is empty", UserWarning)
        return None

    if out_df["OFI"].isna().all():
        warnings.warn(f"{filename}: OFI series is all NaN", UserWarning)
        return None

    return out_df


def compute_slope(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute slope of the order book as in "Della Vedova, Gao, Grant and Westerholm (working paper)".

    Args:
        lob_df: DataFrame with LOB data
        filename: Name of the file being processed
        lob_type: Type of the LOB ("futures" or "ctd")
        context: dict containing keys:
            - 'LOB level' the deepest level to include
            - 'Side' with the side to compute the slope for ("bid" or "ask")

    Returns:
        DataFrame with slope series
    """
    context = context or {}

    if "LOB level" not in context:
        warnings.warn(f"{filename}: Context 'LOB level' is missing", UserWarning)
        return None

    if "Side" not in context:
        warnings.warn(f"{filename}: Context 'Side' is missing", UserWarning)
        return None

    lvl = context.get("LOB level")
    if lvl not in range(1, config.LEVELS + 1):
        warnings.warn(
            f"{filename}: Context 'LOB level' ({lvl}) out of range 1 - {config.LEVELS}", UserWarning
        )
        return None

    side = context.get("Side")
    if side not in {"bid", "ask"}:
        warnings.warn(
            f"{filename}: Context 'Side' ({side}) is invalid. Expected 'bid' or 'ask'", UserWarning
        )
        return None

    best_bid_col = config.BID_PRICE_COL_TEMPLATE.format(lvl=1)
    best_ask_col = config.ASK_PRICE_COL_TEMPLATE.format(lvl=1)
    if side == "bid":
        price_col = config.BID_PRICE_COL_TEMPLATE.format(lvl=lvl)
        size_cols = [
            config.BID_SIZE_COL_TEMPLATE.format(lvl=i) for i in range(1, config.LEVELS + 1)
        ]
    if side == "ask":
        price_col = config.ASK_PRICE_COL_TEMPLATE.format(lvl=lvl)
        size_cols = [
            config.ASK_SIZE_COL_TEMPLATE.format(lvl=i) for i in range(1, config.LEVELS + 1)
        ]

    required_cols = {best_bid_col, best_ask_col, price_col} | set(size_cols)
    if not required_cols.issubset(lob_df.columns):
        warnings.warn(
            f"{filename}: The following required columns are missing: {', '.join(required_cols - set(lob_df.columns))}",
            UserWarning,
        )
        return None

    best_bid = lob_df[best_bid_col]
    best_ask = lob_df[best_ask_col]
    mid_price = (best_bid + best_ask) / 2
    sizes = lob_df[size_cols]
    price = lob_df[price_col]

    total_size = sizes.sum(axis=1)
    price_diff = np.abs(price - mid_price)
    slope = total_size / price_diff

    if slope.empty:
        warnings.warn(f"{filename}: Slope series is empty", UserWarning)
        return None

    if slope.isna().all():
        warnings.warn(f"{filename}: Slope series is all NaN", UserWarning)
        return None

    if (slope == np.inf).any() or (slope == -np.inf).any():
        warnings.warn(
            f"{filename}: Slope series contains infinite values. Replaced with NaNs",
            UserWarning,
        )
        slope.replace([np.inf, -np.inf], np.nan, inplace=True)

    if slope.isna().all():
        warnings.warn(
            f"{filename}: Slope series is all NaN after replacing inf with NaN",
            UserWarning,
        )
        return None

    out_df = pd.DataFrame(
        {
            "Slope": slope,
        },
        index=lob_df.index,
    )
    out_df.index.name = lob_df.index.name

    return out_df
