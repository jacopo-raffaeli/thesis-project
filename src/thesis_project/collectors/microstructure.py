# TODO: Add warning statements where necessary
# TODO: Add minimal comments where needed

import pandas as pd


def collect_spread_series(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
):
    """
    Return timestamped spread series for one file.

    TODO: Add variables description
    TODO: Add output description
    """
    required_cols = {"L1-BidPrice", "L1-AskPrice"}
    if not required_cols.issubset(lob_df.columns):
        print(f"Warning: {filename} missing spread columns.")
        return None

    spread = (lob_df["L1-AskPrice"] - lob_df["L1-BidPrice"]).dropna()
    if spread.empty:
        return None

    return pd.DataFrame(
        {
            "filename": filename,
            "lob_type": lob_type,
            "timestamp": spread.index,
            "date": pd.to_datetime(spread.index).date,
            "spread": spread.to_numpy(),
        }
    )


def collect_mid_returns_series(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
):
    """Return timestamped mid and return series for one file."""
    if "MidPrice" not in lob_df.columns:
        print(f"Warning: {filename} missing MidPrice column.")
        return None

    mid = lob_df["MidPrice"].dropna()
    if mid.empty:
        return None

    returns = mid.pct_change().dropna()
    if returns.empty:
        return None

    aligned_mid = mid.loc[returns.index]
    return pd.DataFrame(
        {
            "filename": filename,
            "lob_type": lob_type,
            "timestamp": returns.index,
            "date": pd.to_datetime(returns.index).date,
            "mid": aligned_mid.to_numpy(),
            "ret": returns.to_numpy(),
        }
    )


def collect_depth_imbalance_series(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
):
    """Return timestamped depth and imbalance series for one file."""
    bid_size_cols = [c for c in lob_df.columns if c.startswith("L") and c.endswith("-BidSize")]
    ask_size_cols = [c for c in lob_df.columns if c.startswith("L") and c.endswith("-AskSize")]

    if not bid_size_cols or not ask_size_cols:
        print(f"Warning: {filename} missing depth columns.")
        return None

    bid_depth = pd.Series(
        lob_df[bid_size_cols].to_numpy(dtype=float).sum(axis=1),
        index=lob_df.index,
    )
    ask_depth = pd.Series(
        lob_df[ask_size_cols].to_numpy(dtype=float).sum(axis=1),
        index=lob_df.index,
    )

    denom = (bid_depth + ask_depth).replace(0, pd.NA)
    imbalance = ((bid_depth - ask_depth) / denom).dropna()
    if imbalance.empty:
        return None

    aligned_bid = bid_depth.loc[imbalance.index]
    aligned_ask = ask_depth.loc[imbalance.index]

    return pd.DataFrame(
        {
            "filename": filename,
            "lob_type": lob_type,
            "timestamp": imbalance.index,
            "date": pd.to_datetime(imbalance.index).date,
            "bid_depth": aligned_bid.to_numpy(),
            "ask_depth": aligned_ask.to_numpy(),
            "imbalance": imbalance.to_numpy(),
        }
    )
