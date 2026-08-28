import pandas as pd

from thesis_project import utils


def _validate_series(s: pd.Series):
    if not isinstance(s, pd.Series):
        raise TypeError("")

    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("")

    utils.checks.is_sampled_at_freq(s)


def compute_gross_basis(*, price_ctd: pd.Series, price_fut: pd.Series, cf: pd.Series) -> pd.Series:
    _validate_series(price_ctd)
    _validate_series(price_fut)
    _validate_series(cf)

    aligned = pd.concat(
        [price_ctd.rename("ctd"), price_fut.rename("fut")],
        axis=1,
        join="inner",
    )

    cf = align_cf(aligned, cf)

    return price_ctd - cf * price_fut


def compute_net_basis(): ...


def compute_tradable_basis(
    *,
    price_ctd: pd.Series,
    price_fut: pd.Series,
    cf: float | pd.Series,
    notional_ctd: float,
    notional_fut: float,
    n_contracts_ctd: float,
): ...


def align_cf(prices: pd.DataFrame, daily_cf: pd.Series) -> pd.Series:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(prices.index).__name__!r}")

    if not isinstance(daily_cf.index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(daily_cf.index).__name__!r}")

    cf = daily_cf.copy()
    cf.index = pd.to_datetime(cf.index).normalize()

    dates = prices.index.tz_localize(None).normalize()
    aligned = pd.Series(cf.reindex(dates).to_numpy(), index=prices.index, name="cf")

    if aligned.isna().any():
        missing_dates = dates[aligned.isna()].unique()
        raise ValueError(f"Missing conversion factor for dates: {missing_dates.tolist()}")

    return aligned
