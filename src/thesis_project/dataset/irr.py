from typing import Literal

import pandas as pd

from thesis_project import config, utils
from thesis_project.utils import misc
from thesis_project.utils.checks import check_series

# TODO:
# - Handle ex-coupon dates correctly.
# - Verify/document whether the current delivery-date convention is the desired one.


def _get_coupon_schedule(
    first_coupon_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
    coupon_freq: int,
) -> pd.DatetimeIndex:
    """
    Generate coupon payment dates from first coupon to maturity.

    The current implementation assumes a regular coupon schedule with
    ``12 / coupon_freq`` months between payments.
    """
    if coupon_freq <= 0:
        raise ValueError("Coupon frequency must be positive")

    if 12 % coupon_freq != 0:
        raise ValueError("Coupon frequency must divide 12")

    if first_coupon_date > maturity_date:
        raise ValueError("First coupon date cannot be after maturity date")

    step_months = 12 // coupon_freq

    return pd.date_range(
        start=first_coupon_date,
        end=maturity_date,
        freq=pd.DateOffset(months=step_months),
    )


def _get_prev_next_coupon(
    coupon_dates: pd.DatetimeIndex,
    ref_date: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the previous and next coupon dates relative to ``ref_date``.

    A coupon occurring exactly on ``ref_date`` is considered the previous
    coupon date.
    """
    prev_coupons = coupon_dates[coupon_dates <= ref_date]
    next_coupons = coupon_dates[coupon_dates > ref_date]

    prev_coupon = prev_coupons.max() if len(prev_coupons) else pd.NaT
    next_coupon = next_coupons.min() if len(next_coupons) else pd.NaT

    assert isinstance(prev_coupon, pd.Timestamp)
    assert isinstance(next_coupon, pd.Timestamp)
    return prev_coupon, next_coupon


def _compute_accrued_interest(
    coupon_rate: float,
    coupon_freq: int,
    prev_coupon_date: pd.Timestamp,
    next_coupon_date: pd.Timestamp,
    val_date: pd.Timestamp,
    notional: float,
) -> float:
    """
    Compute accrued interest using the current ACT/ACT-style convention.

    Accrued interest is computed as the fraction of the coupon period elapsed
    at ``val_date`` multiplied by the coupon amount for the period.

    The convention is retained from the original implementation.
    """
    if pd.isna(prev_coupon_date) or pd.isna(next_coupon_date):
        return 0.0

    total_days = (next_coupon_date - prev_coupon_date).days
    elapsed_days = (val_date - prev_coupon_date).days

    if total_days <= 0:
        return 0.0

    accrual_fraction = elapsed_days / total_days

    return notional * (coupon_rate / coupon_freq) * accrual_fraction


def _get_delivery_date(
    date: pd.Timestamp,
    delivery_dates: pd.DatetimeIndex,
) -> pd.Timestamp:
    """
    Return the first delivery date strictly after ``date``.

    This preserves the convention used by the original implementation.
    """
    valid_dates = delivery_dates[delivery_dates > date]

    if len(valid_dates) == 0:
        raise ValueError(f"No valid delivery date found after {date.date()}")

    return valid_dates[0]


def _get_bond_metadata(
    isin: str,
    ctd_metadata: pd.DataFrame,
) -> pd.Series:
    """
    Return CTD metadata corresponding to an ISIN.
    """
    required_columns = {
        "ISIN",
        "Coupon Annual Rate",
        "Coupon Frequency",
        "First Coupon Date",
        "Maturity Date",
    }

    missing_columns = required_columns - set(ctd_metadata.columns)
    if missing_columns:
        raise ValueError(f"CTD metadata is missing required columns: " f"{sorted(missing_columns)}")

    matches = ctd_metadata.loc[ctd_metadata["ISIN"] == isin]

    if matches.empty:
        raise ValueError(f"Bond metadata not found for ISIN {isin!r}")

    if len(matches) > 1:
        raise ValueError(f"Multiple metadata rows found for ISIN {isin!r}")

    return matches.iloc[0]


def _compute_irr_for_day(
    *,
    fut_price: pd.Series,
    ctd_price: pd.Series,
    cf: float,
    isin: str,
    date: pd.Timestamp,
    delivery_date: pd.Timestamp,
    ctd_metadata: pd.DataFrame,
    notional: float,
) -> pd.Series:
    """
    Compute the IRR series for one trading day.

    All parameters other than the futures and CTD prices are constant within
    the trading day, allowing the calculation to be vectorized.
    """
    bond_metadata = _get_bond_metadata(
        isin=isin,
        ctd_metadata=ctd_metadata,
    )

    coupon_rate = float(bond_metadata["Coupon Annual Rate"])
    coupon_freq = int(bond_metadata["Coupon Frequency"])
    first_coupon_date = pd.Timestamp(bond_metadata["First Coupon Date"])
    maturity_date = pd.Timestamp(bond_metadata["Maturity Date"])

    coupon_schedule = _get_coupon_schedule(
        first_coupon_date=first_coupon_date,
        maturity_date=maturity_date,
        coupon_freq=coupon_freq,
    )

    coupon_dates_relevant = coupon_schedule[
        (coupon_schedule > date) & (coupon_schedule <= delivery_date)
    ]

    coupon_amount = notional * coupon_rate / coupon_freq

    sum_coupons = coupon_amount * len(coupon_dates_relevant)

    sum_weighted_coupons = sum(
        coupon_amount * (delivery_date - coupon_date).days / 360.0
        for coupon_date in coupon_dates_relevant
    )

    T = (delivery_date - date).days / 360.0

    prev_coupon_val, next_coupon_val = _get_prev_next_coupon(
        coupon_dates=coupon_schedule,
        ref_date=date,
    )

    ai_val = _compute_accrued_interest(
        coupon_rate=coupon_rate,
        coupon_freq=coupon_freq,
        prev_coupon_date=prev_coupon_val,
        next_coupon_date=next_coupon_val,
        val_date=date,
        notional=notional,
    )

    prev_coupon_del, next_coupon_del = _get_prev_next_coupon(
        coupon_dates=coupon_schedule,
        ref_date=delivery_date,
    )

    ai_del = _compute_accrued_interest(
        coupon_rate=coupon_rate,
        coupon_freq=coupon_freq,
        prev_coupon_date=prev_coupon_del,
        next_coupon_date=next_coupon_del,
        val_date=delivery_date,
        notional=notional,
    )

    dirty_price = ctd_price + ai_val
    invoice_price = fut_price * cf + ai_del

    numerator = invoice_price - (dirty_price - sum_coupons)

    denominator = dirty_price * T - sum_weighted_coupons

    irr = numerator / denominator

    return irr.rename("irr")


def _compute_irr(
    *,
    fut_price: pd.Series,
    ctd_price: pd.Series,
    cf: pd.Series,
    isin: pd.Series,
    delivery_dates: pd.DatetimeIndex,
    ctd_metadata: pd.DataFrame,
    notional: float = 100,
) -> pd.Series:
    """
    Compute implied repo rate from aligned futures and CTD price series.

    The calculation preserves the original IRR formula. Price observations
    are aligned on their common timestamps, while conversion factors and
    CTD metadata are handled separately on a daily basis.

    The IRR calculation is vectorized within each trading day.
    """
    check_series(fut_price)
    check_series(ctd_price)
    check_series(cf)
    check_series(isin)

    if not isinstance(delivery_dates, pd.DatetimeIndex):
        raise TypeError(
            "Expected delivery_dates to be a DatetimeIndex, "
            f"got {type(delivery_dates).__name__!r}"
        )

    if not isinstance(ctd_metadata, pd.DataFrame):
        raise TypeError(
            f"Expected ctd_metadata to be a DataFrame, " f"got {type(ctd_metadata).__name__!r}"
        )

    # Align the two price series first. The IRR is only defined where both
    # instruments have an observation.
    prices = pd.concat(
        [
            fut_price.rename("fut"),
            ctd_price.rename("ctd"),
        ],
        axis=1,
        join="inner",
    )

    if prices.empty:
        raise ValueError("Futures and CTD price series have no common timestamps")

    # Align daily CF to the common price timestamps.
    aligned_cf = misc.align_cf(
        prices=prices,
        daily_cf=cf,
    )

    prices["cf"] = aligned_cf

    # ISIN is daily information. Reuse the same date-alignment convention as
    # conversion factors.
    aligned_isin = misc.align_cf(
        prices=prices,
        daily_cf=isin,
    )
    prices["isin"] = aligned_isin

    results = []

    assert isinstance(prices.index, pd.DatetimeIndex)
    dates = prices.index.tz_localize(None).normalize()
    for date, day_prices in prices.groupby(dates):
        day_date = pd.Timestamp(date)

        cf_values = day_prices["cf"].unique()
        if len(cf_values) != 1:
            raise ValueError(f"Multiple conversion factors found for {day_date.date()}")

        isin_values = day_prices["isin"].unique()
        if len(isin_values) != 1:
            raise ValueError(f"Multiple CTD ISINs found for {day_date.date()}")

        cf_value = float(cf_values[0])
        isin_value = str(isin_values[0])

        delivery_date = _get_delivery_date(
            date=day_date,
            delivery_dates=delivery_dates,
        )

        day_irr = _compute_irr_for_day(
            fut_price=day_prices["fut"],
            ctd_price=day_prices["ctd"],
            cf=cf_value,
            isin=isin_value,
            date=day_date,
            delivery_date=delivery_date,
            ctd_metadata=ctd_metadata,
            notional=notional,
        )

        results.append(day_irr)

    return pd.concat(results).sort_index().rename("irr")


def _load_price_series(
    ticker: config.FutTicker,
    *,
    directory: str,
    filename: str,
) -> pd.Series:
    path = (config.DATA_PRO_DIR / ticker / "microstructure" / directory / filename).with_suffix(
        ".parquet"
    )

    if not path.exists():
        raise ValueError(f"Price path does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Price path is not a file: {path}")

    price = pd.read_parquet(path).squeeze("columns")

    if not isinstance(price, pd.Series):
        raise TypeError(f"Expected a Series in {path}")

    return price


def _compute_irr_from_prices(
    *,
    ticker: config.FutTicker,
    fut_price_filename: str,
    ctd_price_filename: str,
    price_directory: str,
) -> pd.Series:
    fut_price = _load_price_series(
        ticker=ticker,
        directory=price_directory,
        filename=fut_price_filename,
    )

    ctd_price = _load_price_series(
        ticker=ticker,
        directory=price_directory,
        filename=ctd_price_filename,
    )

    daily_cf = utils.io.load_cf(ticker, "daily_cf.csv")

    cf = daily_cf["CF"]
    isin = daily_cf["ISIN"]

    delivery_dates = utils.io.load_fut_delivery_dates(ticker)
    ctd_metadata = utils.io.load_ctd_metadata(ticker)

    return _compute_irr(
        fut_price=fut_price,
        ctd_price=ctd_price,
        cf=cf,
        isin=isin,
        delivery_dates=delivery_dates,
        ctd_metadata=ctd_metadata,
    )


def compute_irr_mid(ticker: config.FutTicker) -> pd.Series:
    return _compute_irr_from_prices(
        ticker=ticker,
        price_directory="mid_price",
        ctd_price_filename="ctd_mid_price",
        fut_price_filename="fut_mid_price",
    )


def compute_irr_long(ticker: config.FutTicker) -> pd.Series:
    """
    Compute IRR using prices corresponding to entering a long basis position.

    This uses CTD ask and futures bid prices.
    """
    return _compute_irr_from_prices(
        ticker=ticker,
        price_directory="prices",
        ctd_price_filename="ctd_ask_price_1",
        fut_price_filename="fut_bid_price_1",
    )


def compute_irr_short(ticker: config.FutTicker) -> pd.Series:
    """
    Compute IRR using prices corresponding to entering a short basis position.

    This uses CTD bid and futures ask prices.
    """
    return _compute_irr_from_prices(
        ticker=ticker,
        price_directory="prices",
        ctd_price_filename="ctd_bid_price_1",
        fut_price_filename="fut_ask_price_1",
    )


IrrType = Literal[
    "mid",
    "long",
    "short",
]


def compute_irr(
    ticker: config.FutTicker,
    *,
    mode: IrrType,
) -> pd.Series:
    match mode:
        case "mid":
            return compute_irr_mid(ticker)

        case "long":
            return compute_irr_long(ticker)

        case "short":
            return compute_irr_short(ticker)

        case _:
            raise ValueError(f"Unexpected IrrType: {mode!r}")
