import warnings
from typing import Any, Dict, Optional

import pandas as pd

from thesis_project import config


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

    mid = lob_df[ask_col] + lob_df[bid_col] / 2
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
    micro = num / den
    if micro.empty:
        warnings.warn(f"{filename}: Mid price series is empty", UserWarning)
        return None

    out_df = pd.DataFrame(
        {
            "Micro Price": micro,
        },
        index=lob_df.index,
    )
    out_df.index.name = lob_df.index.name

    return out_df
