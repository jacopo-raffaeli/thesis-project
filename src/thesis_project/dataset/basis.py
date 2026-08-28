from typing import Literal

import pandas as pd

from thesis_project import config, utils


def _validate_series(s: pd.Series):
    if not isinstance(s, pd.Series):
        raise TypeError("")

    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("")

    utils.checks.is_sampled_at_freq(s)


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


def frac_fut_contracts(
    cf: pd.Series,
    ctd_contracts: int,
    ctd_face_value: float = config.BTP.contract_size,
    fut_face_value: float = config.FBTP.contract_size,
) -> pd.Series:
    return (cf * ctd_contracts * (ctd_face_value / fut_face_value)).rename("fut_contracts")


def round_fut_contracts(
    fut_contracts: pd.Series,
) -> pd.Series:
    fut_contracts = fut_contracts.round().astype(int)
    if (fut_contracts == 0).any():
        raise ValueError("Position size produces zero futures contracts")

    return fut_contracts.rename("fut_contracts")


def compute_eff_cf(
    ctd_contracts: int,
    fut_contracts: pd.Series,
    ctd_face_value: float = config.BTP.contract_size,
    fut_face_value: float = config.FBTP.contract_size,
) -> pd.Series:
    return (fut_contracts * fut_face_value / (ctd_contracts * ctd_face_value)).rename(
        "effective_cf"
    )


def _compute_gross_basis(*, ctd_price: pd.Series, fut_price: pd.Series, cf: pd.Series) -> pd.Series:
    _validate_series(ctd_price)
    _validate_series(fut_price)
    _validate_series(cf)

    aligned = pd.concat(
        [ctd_price.rename("ctd"), fut_price.rename("fut")],
        axis=1,
        join="inner",
    )

    cf = align_cf(aligned, cf)

    return ctd_price - cf * fut_price


def compute_gross_basis_mid(ticker: config.FutTicker) -> pd.Series:
    root = config.DATA_PRO_DIR / ticker / "microstructure" / "mid_price"
    ctd_filename = "ctd_mid_price"
    fut_filename = "fut_mid_price"
    ctd_path = root / ctd_filename
    fut_path = root / fut_filename

    if not ctd_path.exists():
        raise ValueError("The ctd mid price path do not exists")

    if not ctd_path.is_file():
        raise ValueError("The ctd mid price file does not exists")

    if not fut_path.exists():
        raise ValueError("The fut mid price path do not exists")

    if not fut_path.is_file():
        raise ValueError("The fut mid price file does not exists")

    ctd_mid_price = pd.read_parquet(ctd_path).squeeze("columns")
    fut_mid_price = pd.read_parquet(fut_path).squeeze("columns")
    assert isinstance(ctd_mid_price, pd.Series)
    assert isinstance(fut_mid_price, pd.Series)

    cf = utils.io.load_cf(ticker, "daily_cf.csv")["CF"]

    return _compute_gross_basis(ctd_price=ctd_mid_price, fut_price=fut_mid_price, cf=cf)


def compute_gross_basis_ask(ticker: config.FutTicker) -> pd.Series:
    root = config.DATA_PRO_DIR / ticker / "microstructure" / "prices"
    ctd_filename = "ctd_ask_price_1"
    fut_filename = "fut_bid_price_1"
    ctd_path = root / ctd_filename
    fut_path = root / fut_filename

    if not ctd_path.exists():
        raise ValueError("The ctd best ask price path do not exists")

    if not ctd_path.is_file():
        raise ValueError("The ctd best ask price file does not exists")

    if not fut_path.exists():
        raise ValueError("The fut best bid price path do not exists")

    if not fut_path.is_file():
        raise ValueError("The fut best bid price file does not exists")

    ctd_ask_price = pd.read_parquet(ctd_path).squeeze("columns")
    fut_bid_price = pd.read_parquet(fut_path).squeeze("columns")
    assert isinstance(ctd_ask_price, pd.Series)
    assert isinstance(fut_bid_price, pd.Series)

    cf = utils.io.load_cf(ticker, "daily_cf.csv")["CF"]

    return _compute_gross_basis(ctd_price=ctd_ask_price, fut_price=fut_bid_price, cf=cf)


def compute_gross_basis_bid(ticker: config.FutTicker) -> pd.Series:
    root = config.DATA_PRO_DIR / ticker / "microstructure" / "prices"
    ctd_filename = "ctd_bid_price_1"
    fut_filename = "fut_ask_price_1"
    ctd_path = root / ctd_filename
    fut_path = root / fut_filename

    if not ctd_path.exists():
        raise ValueError("The ctd best bid price path do not exists")

    if not ctd_path.is_file():
        raise ValueError("The ctd best bid price file does not exists")

    if not fut_path.exists():
        raise ValueError("The fut best ask price path do not exists")

    if not fut_path.is_file():
        raise ValueError("The fut best ask price file does not exists")

    ctd_bid_price = pd.read_parquet(ctd_path).squeeze("columns")
    fut_ask_price = pd.read_parquet(fut_path).squeeze("columns")
    assert isinstance(ctd_bid_price, pd.Series)
    assert isinstance(fut_ask_price, pd.Series)

    cf = utils.io.load_cf(ticker, "daily_cf.csv")["CF"]

    return _compute_gross_basis(ctd_price=ctd_bid_price, fut_price=fut_ask_price, cf=cf)


GrossBasisType = Literal[
    "mid",
    "bid",
    "ask",
]


def compute_gross_basis(*, ticker: config.FutTicker, mode: GrossBasisType) -> pd.Series:
    match mode:
        case "mid":
            return compute_gross_basis_mid(ticker)

        case "ask":
            return compute_gross_basis_ask(ticker)

        case "bid":
            return compute_gross_basis_bid(ticker)

        case _:
            raise ValueError(f"Unexpected GrossBasisType: {mode!r}")


def compute_tradable_basis(): ...


def compute_net_basis(): ...
