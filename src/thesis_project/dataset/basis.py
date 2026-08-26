import pandas as pd


def compute_gross_basis(
    *, price_ctd: pd.Series, price_fut: pd.Series, cf: float | pd.Series
) -> pd.Series:
    if not isinstance(price_ctd, pd.Series):
        raise TypeError("")

    if not isinstance(price_fut, pd.Series):
        raise TypeError("")

    if not isinstance(price_ctd.index, pd.DatetimeIndex):
        raise TypeError("")

    if not isinstance(price_fut.index, pd.DatetimeIndex):
        raise TypeError("")

    if not price_ctd.index.equals(price_fut.index):
        raise ValueError

    if isinstance(cf, pd.Series):
        if not isinstance(cf.index, pd.DatetimeIndex):
            raise TypeError("")

        if not price_ctd.index.equals(cf.index):
            raise ValueError("")

    return price_ctd - cf * price_fut


def compute_net_basis(): ...


def compute_tradable_basis(): ...
