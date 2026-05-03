import warnings
from typing import Dict, Optional

import pandas as pd


def _get_coupon_schedule(
    first_coupon_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
    coupon_freq: int,
) -> pd.DatetimeIndex:
    """
    Generate coupon payment dates from first coupon to maturity.

    Args:
        first_coupon_date: Date of first coupon payment
        maturity_date: Bond maturity date
        coupon_freq: Coupon frequency (e.g., 2 for semi-annual)

    Returns:
        DatetimeIndex of coupon dates
    """
    step_months = 12 // coupon_freq
    return pd.date_range(
        start=first_coupon_date,
        end=maturity_date,
        freq=pd.DateOffset(months=step_months),
    )


def _get_prev_next_coupon(
    coupon_dates: pd.DatetimeIndex,
    ref_date: pd.Timestamp,
) -> tuple:
    """
    Find previous and next coupon dates relative to reference date.

    Args:
        coupon_dates: DatetimeIndex of all coupon dates
        ref_date: Reference date

    Returns:
        Tuple (prev_coupon_date, next_coupon_date), may contain NaT
    """
    prev_coupons = coupon_dates[coupon_dates <= ref_date]
    next_coupons = coupon_dates[coupon_dates > ref_date]

    prev_c = prev_coupons.max() if len(prev_coupons) > 0 else pd.NaT
    next_c = next_coupons.min() if len(next_coupons) > 0 else pd.NaT

    return prev_c, next_c


def _compute_accrued_interest(
    coupon_rate: float,
    coupon_freq: int,
    prev_coupon_date: pd.Timestamp,
    next_coupon_date: pd.Timestamp,
    val_date: pd.Timestamp,
    notional: float = 100.0,
) -> float:
    """
    Compute accrued interest using day-wise ACT/ACT calculation.

    Args:
        coupon_rate: Annual coupon rate (decimal)
        coupon_freq: Coupon frequency per year
        prev_coupon_date: Previous coupon date
        next_coupon_date: Next coupon date
        val_date: Valuation date
        notional: Bond face value

    Returns:
        Accrued interest (float), or 0.0 if prev/next coupon missing
    """
    if pd.isna(prev_coupon_date) or pd.isna(next_coupon_date):
        return 0.0

    total_days = (next_coupon_date - prev_coupon_date).days
    elapsed_days = (val_date - prev_coupon_date).days

    if total_days <= 0:
        return 0.0

    accrual_frac = elapsed_days / total_days
    return notional * (coupon_rate / coupon_freq) * accrual_frac


def _resample_lob(
    lob_df: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    """
    Resample LOB data to target frequency using last quote.

    Args:
        lob_df: LOB DataFrame indexed by timestamp
        frequency: Pandas frequency string (e.g., "1h", "1D")

    Returns:
        Resampled LOB DataFrame
    """
    return lob_df.resample(frequency).last()


def _compute_repo_at_timestamp(
    fut_price: float,
    ctd_price: float,
    coupon_rate: float,
    coupon_freq: int,
    cf: float,
    coupon_dates_relevant: pd.DatetimeIndex,
    ai_val: float,
    ai_del: float,
    T: float,
    delivery_date: pd.Timestamp,
    notional: float = 100.0,
) -> float:
    """
    Compute implied repo rate using closed formula.

    R = (B - A0 + sum(K)) / (A0 * T - sum(K * t_i))

    where:
    - A0 = dirty_price at valuation = ctd_price + ai_val
    - B = invoice_price = fut_price * cf + ai_del
    - K = coupon amounts
    - t_i = ACT/360 year fractions from coupon to delivery

    Args:
        fut_price: Futures price
        ctd_price: CTD clean price
        coupon_rate: Annual coupon rate
        coupon_freq: Coupon frequency
        cf: Conversion factor
        coupon_dates_relevant: Coupon dates between valuation and delivery
        ai_val: Accrued interest at valuation
        ai_del: Accrued interest at delivery
        T: Year fraction (ACT/360) from valuation to delivery
        delivery_date: Delivery date (pd.Timestamp or compatible)
        notional: Bond face value

    Returns:
        Implied repo rate (float), or NaN if computation fails
    """
    # Compute dirty price at valuation
    dirty_price_val = ctd_price + ai_val

    # Compute invoice price
    invoice_price = fut_price * cf + ai_del

    # Compute sum of coupon amounts and weighted coupon amounts
    sum_coupons = 0.0
    sum_weighted_coupons = 0.0

    for cpn_date in coupon_dates_relevant:
        coupon_amount = notional * coupon_rate / coupon_freq
        t_i = (delivery_date - cpn_date).days / 360.0
        sum_coupons += coupon_amount
        sum_weighted_coupons += coupon_amount * t_i

    # R = (B - A0 + sum(K)) / (A0*T - sum(K*t_i))
    numerator = invoice_price - (dirty_price_val - sum_coupons)
    denominator = dirty_price_val * T - sum_weighted_coupons

    if denominator == 0.0:
        return float("nan")

    return numerator / denominator


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


def compute_implied_repo(
    fut_df: pd.DataFrame,
    ctd_df: pd.DataFrame,
    context: Optional[Dict],
) -> Optional[pd.DataFrame]:
    """
    Return implied repo rate series at specified frequency.

    Computes R such that: (dirty_price + AI_val) * (1 + R*T) - sum(K*(1+R*t_i)) = F(t) * cf + AI_del

    Args:
        fut_df: DataFrame with futures LOB data, indexed by timestamp, requires MidPrice
        ctd_df: DataFrame with CTD LOB data, indexed by timestamp, requires MidPrice
        context: Dict with required keys:
            - 'CF': daily_cf.csv
            - 'CTD Metadata': ctd_metadata.csv
            - 'FUT Metadata': fut_metadata.csv
            - 'Frequency': resampling frequency for the LOBs
            - 'Notional': Usually 100

    Returns:
        DataFrame with IRR column indexed by timestamp
    """
    context = context or {}

    # Extract date
    fut_date = fut_df.index[0].date()
    ctd_date = ctd_df.index[0].date()
    if fut_date != ctd_date:
        warnings.warn(
            f"Mismatch between LOB futures ({fut_date}) and CTD ({ctd_date}) dates.",
            UserWarning,
        )
        return None
    date = fut_date

    # Validate required columns
    if "MidPrice" not in fut_df.columns or "MidPrice" not in ctd_df.columns:
        warnings.warn(
            f"{date}: Missing MidPrice column in futures or CTD LOB.",
            UserWarning,
        )
        return None

    # Extract context
    if "CF" not in context:
        warnings.warn(f"{date}: Context 'CF' is missing", UserWarning)
        return None
    cf_series = context["CF"]

    if "CTD Metadata" not in context:
        warnings.warn(f"{date}: Context 'CTD Metadata' is missing", UserWarning)
        return None
    ctd_meta_df = context.get("CTD Metadata")

    if "FUT Metadata" not in context:
        warnings.warn(f"{date}: Context 'FUT Metadata' is missing", UserWarning)
        return None
    fut_meta_df = context.get("FUT Metadata")

    if "Frequency" not in context:
        warnings.warn(f"{date}: Context 'Frequency' is missing", UserWarning)
        return None
    frequency = context.get("Frequency")

    if "Notional" not in context:
        warnings.warn(f"{date}: Context 'Notional' is missing", UserWarning)
        return None
    notional = context.get("Notional")

    # Parse metadata date columns
    if not isinstance(fut_meta_df["Delivery Date"].iloc[0], pd.Timestamp):  # type: ignore
        fut_meta_df = fut_meta_df.copy()  # type: ignore
        fut_meta_df["Delivery Date"] = pd.to_datetime(fut_meta_df["Delivery Date"])

    if not isinstance(ctd_meta_df["First Coupon Date"].iloc[0], pd.Timestamp):  # type: ignore
        ctd_meta_df = ctd_meta_df.copy()  # type: ignore
        ctd_meta_df["First Coupon Date"] = pd.to_datetime(ctd_meta_df["First Coupon Date"])
        ctd_meta_df["Maturity Date"] = pd.to_datetime(ctd_meta_df["Maturity Date"])

    # Extract delivery date for this LOB date
    # NOTE: We consider the very next delivery date
    delivery_dates_valid = fut_meta_df["Delivery Date"].dropna()  # type: ignore
    delivery_dates_valid = delivery_dates_valid[delivery_dates_valid > pd.Timestamp(date)]
    if delivery_dates_valid.empty:
        warnings.warn(
            f"{date}: No valid delivery date found.",
            UserWarning,
        )
        return None

    delivery_date = pd.Timestamp(delivery_dates_valid.iloc[0]).normalize()

    # Extract ISIN for this date
    try:
        isin = cf_series.loc[pd.Timestamp(date), "ISIN"]  # type: ignore
    except KeyError:
        warnings.warn(
            f"{date}: Missing ISIN in daily_cf.csv.",
            UserWarning,
        )
        return None

    # Extract bond metadata for this ISIN
    bond_meta = ctd_meta_df[ctd_meta_df["ISIN"] == isin]  # type: ignore
    if bond_meta.empty:
        warnings.warn(
            f"{date}: Bond metadata not found for ISIN {isin}.",
            UserWarning,
        )
        return None

    bond_meta = bond_meta.iloc[0]
    coupon_rate = float(bond_meta["Coupon Annual Rate"])
    coupon_freq = int(bond_meta["Coupon Frequency"])
    first_coupon_date = pd.Timestamp(bond_meta["First Coupon Date"])
    maturity_date = pd.Timestamp(bond_meta["Maturity Date"])

    # Extract CF for this date
    try:
        cf_value = float(cf_series.loc[pd.Timestamp(date), "CF"])  # type: ignore
    except KeyError:
        warnings.warn(
            f"{date}: Missing conversion factor.",
            UserWarning,
        )
        return None

    # Build coupon schedule
    coupon_schedule = _get_coupon_schedule(first_coupon_date, maturity_date, coupon_freq)

    # Get relevant coupons (between LOB date and delivery date)
    # NOTE: Actually I should consider ex-coupon dates
    coupon_dates_relevant = coupon_schedule[
        (coupon_schedule > pd.Timestamp(date)) & (coupon_schedule <= delivery_date)
    ]

    # Compute year fraction (ACT/360)
    T = (delivery_date - pd.Timestamp(date)).days / 360.0

    # Compute accrued interests
    prev_cpn_val, next_cpn_val = _get_prev_next_coupon(coupon_schedule, pd.Timestamp(date))
    ai_val = _compute_accrued_interest(
        coupon_rate,
        coupon_freq,
        prev_cpn_val,
        next_cpn_val,
        pd.Timestamp(date),
        notional,  # type: ignore
    )

    prev_cpn_del, next_cpn_del = _get_prev_next_coupon(coupon_schedule, delivery_date)
    ai_del = _compute_accrued_interest(
        coupon_rate,
        coupon_freq,
        prev_cpn_del,
        next_cpn_del,
        delivery_date,
        notional,  # type: ignore
    )

    # Resample LOBs
    # NOTE: This can be probably handled better
    if frequency != "1s":
        fut_lob_resampled = _resample_lob(fut_df, frequency)  # type: ignore
        ctd_lob_resampled = _resample_lob(ctd_df, frequency)  # type: ignore
    else:
        fut_lob_resampled = fut_df.copy()
        ctd_lob_resampled = ctd_df.copy()

    # Align resampled LOBs
    aligned = pd.concat(
        [fut_lob_resampled[["MidPrice"]], ctd_lob_resampled[["MidPrice"]]],
        axis=1,
        join="inner",
        keys=["fut", "ctd"],
    )

    if aligned.empty:
        warnings.warn(
            f"{date}: No overlapping timestamps after resampling to {frequency}.",
            UserWarning,
        )
        return None

    # Actual IRR computation
    results = []

    for timestamp, row in aligned.iterrows():
        fut_price = row[("fut", "MidPrice")]
        ctd_price = row[("ctd", "MidPrice")]

        # NaN if prices missing
        if pd.isna(fut_price) or pd.isna(ctd_price):
            results.append(float("nan"))
            continue

        # Compute implied repo using closed formula
        repo_rate = _compute_repo_at_timestamp(
            fut_price=fut_price,
            ctd_price=ctd_price,
            coupon_rate=coupon_rate,
            coupon_freq=coupon_freq,
            cf=cf_value,
            coupon_dates_relevant=coupon_dates_relevant,
            ai_val=ai_val,
            ai_del=ai_del,
            T=T,
            delivery_date=delivery_date,
            notional=notional,  # type: ignore
        )
        results.append(repo_rate)

    # Build output DataFrame
    out_df = pd.DataFrame(
        {"IRR": results},
        index=aligned.index.rename("timestamp"),
    )

    return out_df
