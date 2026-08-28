import warnings
from typing import Dict, Optional

import pandas as pd


def compute_gross_basis(
    fut_df: pd.DataFrame,
    ctd_df: pd.DataFrame,
    context: Optional[Dict],
) -> Optional[pd.DataFrame]:
    """
    Return timestamped basis series after alignment.
    Basis is computed as ctd_mid - cf * fut_mid, where cf is daily

    Args:
        fut_df: DataFrame with futures LOB data
        ctd_df: DataFrame with CTD LOB data
        context: Dict with required keys:
            - 'CF': daily_cf.csv

    Returns:
        DataFrame with basis series
    """
    context = context or {}

    # Extract date
    fut_date = fut_df.index[0].date()
    ctd_date = ctd_df.index[0].date()
    if fut_date != ctd_date:
        warnings.warn(
            f"Mismatch between LOB futures ({fut_date}) and CTD ({ctd_date}) dates.", UserWarning
        )
        return None
    date = fut_date

    # Extract mid price series
    mid_col = "MidPrice"
    required_cols = {mid_col}
    if not required_cols.issubset(fut_df.columns):
        warnings.warn(
            f"{date}: The following required columns are missing from futures LOB: {', '.join(required_cols - set(fut_df.columns))}",
            UserWarning,
        )
        return None
    if not required_cols.issubset(ctd_df.columns):
        warnings.warn(
            f"{date}: The following required columns are missing from CTD LOB: {', '.join(required_cols - set(ctd_df.columns))}",
            UserWarning,
        )
        return None
    fut_mid = fut_df[mid_col].rename("FUT MidPrice")
    ctd_mid = ctd_df[mid_col].rename("CTD MidPrice")

    # Load conversion factor series
    if "CF" not in context:
        warnings.warn(f"{date}: Context 'CF' is missing", UserWarning)
        return None
    cf_series = context["CF"]

    # Look up cf value by date
    try:
        cf_value = cf_series.loc[pd.to_datetime(date)]["CF"]
    except KeyError:
        warnings.warn(f"{date}: Missing conversion factor.", UserWarning)
        return None

    # Align on timestamps
    aligned = pd.concat([fut_mid, ctd_mid], axis=1, join="inner")
    if aligned.empty:
        warnings.warn(
            f"{date}: No overlapping timestamps between futures and CTD LOBs.", UserWarning
        )
        return None

    # Compute basis
    basis = aligned["CTD MidPrice"] - float(cf_value) * aligned["FUT MidPrice"]

    out_df = pd.DataFrame(
        {
            "Basis": basis,
        },
        index=basis.index.rename("timestamp"),
    )

    return out_df
