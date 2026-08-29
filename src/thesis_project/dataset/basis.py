from typing import Literal

import pandas as pd

from thesis_project import config, utils
from thesis_project.utils.checks import check_series
from thesis_project.utils.misc import align_cf


def _compute_basis(*, ctd_price: pd.Series, fut_price: pd.Series, cf: pd.Series) -> pd.Series:
    check_series(ctd_price)
    check_series(fut_price)

    aligned = pd.concat(
        [ctd_price.rename("ctd"), fut_price.rename("fut")],
        axis=1,
        join="inner",
    )

    cf = align_cf(aligned, cf)

    return aligned["ctd"] - cf * aligned["fut"]


def compute_basis_mid(ticker: config.FutTicker) -> pd.Series:
    root = config.DATA_PRO_DIR / ticker / "microstructure" / "mid_price"
    ctd_filename = "ctd_mid_price"
    fut_filename = "fut_mid_price"
    ctd_path = (root / ctd_filename).with_suffix(".parquet")
    fut_path = (root / fut_filename).with_suffix(".parquet")

    if not ctd_path.is_file():
        raise FileNotFoundError(f"CTD price not found: {ctd_path.relative_to(config.ROOT)!r}")

    if not fut_path.is_file():
        raise FileNotFoundError(f"FUT price not found: {fut_path.relative_to(config.ROOT)!r}")

    ctd_mid_price = pd.read_parquet(ctd_path).squeeze("columns")
    fut_mid_price = pd.read_parquet(fut_path).squeeze("columns")
    assert isinstance(ctd_mid_price, pd.Series)
    assert isinstance(fut_mid_price, pd.Series)

    cf = utils.io.load_cf(ticker, "daily_cf.csv")["CF"]

    return _compute_basis(ctd_price=ctd_mid_price, fut_price=fut_mid_price, cf=cf)


def compute_basis_ask(ticker: config.FutTicker) -> pd.Series:
    root = config.DATA_PRO_DIR / ticker / "microstructure" / "prices"
    ctd_filename = "ctd_ask_price_1"
    fut_filename = "fut_bid_price_1"
    ctd_path = (root / ctd_filename).with_suffix(".parquet")
    fut_path = (root / fut_filename).with_suffix(".parquet")

    if not ctd_path.is_file():
        raise FileNotFoundError(f"CTD price not found: {ctd_path.relative_to(config.ROOT)!r}")

    if not fut_path.is_file():
        raise FileNotFoundError(f"FUT price not found: {fut_path.relative_to(config.ROOT)!r}")

    ctd_ask_price = pd.read_parquet(ctd_path).squeeze("columns")
    fut_bid_price = pd.read_parquet(fut_path).squeeze("columns")
    assert isinstance(ctd_ask_price, pd.Series)
    assert isinstance(fut_bid_price, pd.Series)

    cf = utils.io.load_cf(ticker, "daily_cf.csv")["CF"]

    return _compute_basis(ctd_price=ctd_ask_price, fut_price=fut_bid_price, cf=cf)


def compute_basis_bid(ticker: config.FutTicker) -> pd.Series:
    root = config.DATA_PRO_DIR / ticker / "microstructure" / "prices"
    ctd_filename = "ctd_bid_price_1"
    fut_filename = "fut_ask_price_1"
    ctd_path = (root / ctd_filename).with_suffix(".parquet")
    fut_path = (root / fut_filename).with_suffix(".parquet")

    if not ctd_path.is_file():
        raise FileNotFoundError(f"CTD price not found: {ctd_path.relative_to(config.ROOT)!r}")

    if not fut_path.is_file():
        raise FileNotFoundError(f"FUT price not found: {fut_path.relative_to(config.ROOT)!r}")

    ctd_bid_price = pd.read_parquet(ctd_path).squeeze("columns")
    fut_ask_price = pd.read_parquet(fut_path).squeeze("columns")
    assert isinstance(ctd_bid_price, pd.Series)
    assert isinstance(fut_ask_price, pd.Series)

    cf = utils.io.load_cf(ticker, "daily_cf.csv")["CF"]

    return _compute_basis(ctd_price=ctd_bid_price, fut_price=fut_ask_price, cf=cf)


GrossBasisType = Literal[
    "mid",
    "bid",
    "ask",
]


def compute_basis(ticker: config.FutTicker, *, mode: GrossBasisType) -> pd.Series:
    match mode:
        case "mid":
            return compute_basis_mid(ticker)

        case "ask":
            return compute_basis_ask(ticker)

        case "bid":
            return compute_basis_bid(ticker)

        case _:
            raise ValueError(f"Unexpected GrossBasisType: {mode!r}")


def compute_tradable_basis(): ...


def compute_net_basis(): ...
