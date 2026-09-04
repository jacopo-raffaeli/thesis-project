import datetime
import re
from pathlib import Path
from typing import Literal, get_args

import dataframe_image as dfi
import pandas as pd
import polars as pl

from thesis_project import config


def df_to_png(df: pd.DataFrame, path: Path, filename: str):
    """
    Export a DataFrame to png with dataframe_image library
    It requires chrome.exe on the machine to work

    ## Args:
    * df: DataFrame to export
    * path: file output path
    * filenmae: file output name
    """
    path = path / filename
    path = path.with_suffix(".png")
    dfi.export(df, path, table_conversion="chrome")


def filename_to_date(filename: str) -> datetime.date:
    """
    Extract the date from a LOB parquet filename.

    ## Args:
    * filename: expected format {ctd/fut}_lob_freq_1s_yyyy_mm_dd.parquet

    ## Return
    * datetime.date object
    """
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})\.parquet$", filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")

    year, month, day = map(int, match.groups())
    return datetime.date(year, month, day)


def load_cf(ticker: config.FutTicker, filename: str = "daily_cf.csv") -> pd.DataFrame:
    """
    Load daily conversion factor series \n
    The csv also contains the day index, the ISIN id and CUSIP id

    ## Args:
    * ticker: FutTicker
    * filename: csv filename

    ## Returns:
    * cf: DataFrame containing
        * Index: Date
        * CF: The conversion factor series
        * ISIN: The respective ISIN ids
        * CUSIP: The respective CUSIP ids
    """
    path = config.DATA_RAW_DIR / ticker / filename
    cf = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")

    assert isinstance(cf.index, pd.DatetimeIndex)
    cf.index = cf.index.normalize()

    if not cf.index.is_monotonic_increasing:
        cf = cf.sort_index(ascending=True)

    return cf


def load_ctd_switch_dates(
    ticker: config.FutTicker, filename: str = "ctd_switch.csv"
) -> list[datetime.date]:
    """
    Load ctd bond switch dates \n
    Such dates are the first in which a new bond is considered the cheapest

    ## Args:
    * ticker: FutTicker
    * filename: csv filename

    ## Returns:
    * dates: DatetimeIndex
    """
    path = config.DATA_RAW_DIR / ticker / filename
    dates = pd.read_csv(path, parse_dates=["CTD Switch Date"])["CTD Switch Date"]
    dates = dates.dropna().dt.date.sort_values().to_list()

    return dates


def load_criticalities(
    ticker: config.FutTicker, filename: str = "criticalities.csv"
) -> list[datetime.date]:
    """
    Load default list of critical dates from data/processed/ticker/filename

    ## Args:
    * ticker: FutTicker

    Return:
    * dates: list of datetime.date
    * filename: csv filename
    """
    path = config.DATA_PRO_DIR / ticker / filename
    path = path.with_suffix(".csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    dates = df["Date"].dt.date.drop_duplicates().to_list()

    return dates


def load_extra(ticker: config.FutTicker, filename: str = "extra.csv") -> list[datetime.date]:
    """
    Load extra list of critical dates from data/processed/ticker/filename

    ## Args:
    * ticker: FutTicker

    Return:
    * dates: list of datetime.date
    * filename: csv filename
    """
    path = config.DATA_PRO_DIR / ticker / filename
    path = path.with_suffix(".csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    dates = df["Date"].dt.date.drop_duplicates().to_list()

    return dates


def load_ctd_metadata(
    ticker: config.FutTicker,
    filename: str = "ctd_metadata.csv",
) -> pd.DataFrame:
    """
    Load CTD bond metadata.

    ## Args:
    * ticker: FutTicker
    * filename: CSV filename

    ## Returns:
    * DataFrame containing CTD metadata
    """
    path = config.DATA_RAW_DIR / ticker / filename

    return pd.read_csv(
        path,
        parse_dates=["First Coupon Date", "Maturity Date"],
    )


def load_fut_metadata(
    ticker: config.FutTicker,
    filename: str = "fut_metadata.csv",
) -> pd.DataFrame:
    """
    Load futures metadata.

    ## Args:
    * ticker: FutTicker
    * filename: CSV filename

    ## Returns:
    * DataFrame containing futures metadata
    """
    path = config.DATA_RAW_DIR / ticker / filename

    return pd.read_csv(
        path,
        parse_dates=["Last Trading Date", "Delivery Date"],
    )


def load_fut_rollover_dates(
    ticker: config.FutTicker, filename: str = "fut_metadata.csv"
) -> list[datetime.date]:
    """
    Load future rollover dates \n
    Such dates are the last in which a certain future contract is traded

    ## Args:
    * ticker: FutTicker
    * filename: csv filename

    ## Returns:
    * dates: DatetimeIndex
    """
    dates = load_fut_metadata(ticker)["Last Trading Date"]
    dates = dates.dropna().dt.date.sort_values().to_list()

    return dates


def load_fut_delivery_dates(
    ticker: config.FutTicker, filename: str = "fut_metadata.csv"
) -> list[datetime.date]:
    """
    Load future delivery dates
    \nSuch dates are the ones in which the short part deliver the underlying to
    the long one, usually a couple of day after the respective rollover date

    ## Args:
    * ticker: FutTicker
    * filename: csv filename

    ## Returns:
    * dates: DatetimeIndex
    """
    dates = load_fut_metadata(ticker)["Delivery Date"]
    dates = dates.dropna().dt.date.sort_values().to_list()

    return dates


def list_lob_paths_role(
    *, root: Path = config.DATA_RAW_DIR, ticker: config.FutTicker, role: config.AssetRole
) -> list[Path]:
    """
    Iterate data/raw/ticker/role and collect all the lob paths

    ## Args:
    * root: Data root folder
    * ticker: FutTicker literal
    * role: AssetRole literal

    ## Return:
    * paths: List of lob paths
    """
    paths = []
    root = root / ticker / role

    if not root.exists():
        raise ValueError(f"This path does not exists: {root}")

    if not root.is_dir():
        raise ValueError(f"This path is not a directory: {root}")

    for ydir in sorted(root.iterdir()):
        if not ydir.is_dir():
            continue

        for mdir in sorted(ydir.iterdir()):
            if not mdir.is_dir():
                continue

            for path in sorted(mdir.iterdir()):
                if not path.is_file():
                    continue

                if config.LOB.filename_re.fullmatch(path.name):
                    paths.append(path)

    return sorted(paths)


def list_lob_paths_ticker(
    *, root: Path = config.DATA_RAW_DIR, ticker: config.FutTicker
) -> list[Path]:
    """
    Iterate data/raw/ticker/ and collect all the lob paths for all the roles

    ## Args:
    * root: Data root folder
    * ticker: FutTicker literal

    ## Return:
    * paths: List of lob paths
    """
    paths = []
    for role in get_args(config.AssetRole):
        paths.extend(list_lob_paths_role(root=root, ticker=ticker, role=role))

    return sorted(paths)


def list_lob_paths(
    *,
    root: Path,
    ticker: config.FutTicker,
    role: config.AssetRole | None = None,
) -> list[Path]:
    """
    Iterate data/{raw|processed}/ticker/Optional[role] and collect all the lob paths

    ## Args:
    * root: Data root folder
    * ticker: FutTicker literal
    * role: AssetRole literal

    ## Return:
    * paths: List of lob paths
    """
    if role is None:
        return list_lob_paths_ticker(root=root, ticker=ticker)

    return list_lob_paths_role(root=root, ticker=ticker, role=role)


def load_filtered_parquet(
    path: Path,
    time_window: tuple[datetime.time, datetime.time] | None = None,
    dates_to_exclude: list[datetime.date] | None = None,
) -> pd.Series:
    """"""
    idx = config.LOB.index_name

    if not path.exists():
        raise ValueError("")

    if not path.is_file():
        raise ValueError("")

    if path.suffix != ".parquet":
        raise ValueError(f"Expected suffix '.parquet', got {path.suffix!r}")

    scan = pl.scan_parquet(path)

    if idx not in scan.collect_schema().names():
        raise ValueError(f"Expected {idx} column not found")

    if dates_to_exclude is not None:
        scan = scan.filter(~pl.col(idx).dt.date().is_in(dates_to_exclude))

    if time_window is not None:
        min_time, max_time = time_window
        scan = scan.filter(pl.col(idx).dt.time().is_between(min_time, max_time))

    data = scan.collect().to_pandas().set_index(idx).sort_index()

    if len(data.columns) != 1:
        raise ValueError("")

    data = data.squeeze("columns")

    if not isinstance(data, pd.Series):
        raise TypeError("")

    return data


def load_lob(
    *,
    folder: Literal["raw", "processed"] = "processed",
    ticker: config.FutTicker,
    role: config.AssetRole,
    date: datetime.date | str,
    freq: str = config.LOB.freq,
) -> pd.DataFrame:
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)

    yyyy = str(date.year).zfill(4)
    mm = str(date.month).zfill(2)
    dd = str(date.day).zfill(2)

    filename = config.LOB.FILENAME.format(
        role=role,
        freq=freq,
        yyyy=yyyy,
        mm=mm,
        dd=dd,
    )

    path = config.DATA_DIR / folder / ticker / role / yyyy / mm / filename
    lob = pd.read_parquet(path)

    return lob


def get_run_path(root: Path) -> Path:
    pattern = re.compile(r"^run-(\d{3})$")
    runs = [
        (int(match.group(1)), path)
        for path in root.iterdir()
        if path.is_dir() and (match := pattern.match(path.name))
    ]

    n = max((i for i, _ in runs), default=0) + 1
    if n > 999:
        raise ValueError

    path = root / f"run-{n:03d}"

    return path


def create_run_path(root: Path):
    path = get_run_path(root)
    path.mkdir(parents=True, exist_ok=False)

    return path
