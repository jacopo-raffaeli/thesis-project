from pathlib import Path

import pandas as pd


def load_daily_cf(cf_path: Path) -> pd.Series:
    """
    Load daily conversion factors indexed by date.

    TODO: Add variables description
    TODO: Add output description
    """
    cf_df = pd.read_csv(cf_path)
    if "Date" not in cf_df.columns or "CF" not in cf_df.columns:
        raise ValueError("CF CSV must contain 'Date' and 'CF' columns.")

    cf_df["Date"] = [pd.Timestamp(value).date() for value in cf_df["Date"]]
    return cf_df.set_index("Date")["CF"]


def collect_basis_series(
    fut_df: pd.DataFrame,
    ctd_df: pd.DataFrame,
    cf_value: float,
):
    """Return timestamped basis series after alignment.
    Basis is computed as ctd_mid - cf * fut_mid.

    TODO: Add variables description
    TODO: Add output description
    """
    fut_mid = fut_df["MidPrice"].rename("fut_mid")
    ctd_mid = ctd_df["MidPrice"].rename("ctd_mid")

    aligned = pd.concat([fut_mid, ctd_mid], axis=1, join="inner").dropna()
    if aligned.empty:
        return None

    basis = aligned["ctd_mid"] - float(cf_value) * aligned["fut_mid"]
    return pd.DataFrame(
        {
            "timestamp": basis.index,
            "date": pd.to_datetime(basis.index).date,
            "fut_mid": aligned["fut_mid"].to_numpy(),
            "ctd_mid": aligned["ctd_mid"].to_numpy(),
            "cf": float(cf_value),
            "basis": basis.to_numpy(),
        }
    )


def collect_rollover_convergence_series(
    active_contract_df: pd.DataFrame,
    next_contract_df: pd.DataFrame,
    rollover_ts,
    window: str = "30min",
):
    """
    Return timestamped pre/post rollover spread series between contracts.

    TODO: Add variables description
    TODO: Add output description
    """
    active_mid = active_contract_df["MidPrice"].rename("active_mid")
    next_mid = next_contract_df["MidPrice"].rename("next_mid")

    aligned = pd.concat([active_mid, next_mid], axis=1, join="inner").dropna()
    if aligned.empty:
        return None

    spread = aligned["active_mid"] - aligned["next_mid"]

    rollover_ts = pd.Timestamp(rollover_ts)
    half_window = pd.Timedelta(window)

    pre = spread[(spread.index >= rollover_ts - half_window) & (spread.index < rollover_ts)]
    post = spread[(spread.index >= rollover_ts) & (spread.index <= rollover_ts + half_window)]

    pre_df = pd.DataFrame(
        {
            "timestamp": pre.index,
            "date": pd.to_datetime(pre.index).date,
            "window_side": "pre",
            "spread": pre.to_numpy(),
        }
    )
    post_df = pd.DataFrame(
        {
            "timestamp": post.index,
            "date": pd.to_datetime(post.index).date,
            "window_side": "post",
            "spread": post.to_numpy(),
        }
    )

    result = pd.concat([pre_df, post_df], ignore_index=True)
    return result if not result.empty else None
