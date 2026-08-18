import datetime
import re
from pathlib import Path

import dataframe_image as dfi
import pandas as pd

from thesis_project import config as global_config


def df_to_png(df: pd.DataFrame, path: Path, filename: str):
    """
    Export a DataFrame to png with dataframe_image library
    It requires chrome.exe on the machine to work

    ## Args:
    * df: DataFrame to export
    * path: file output path
    * filenmae: file output name
    """
    dfi.export(df, path / f"{filename}.png", table_conversion="chrome")


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


def load_cf(ticker: global_config.FutTicker, filename: str = "daily_cf.csv") -> pd.DataFrame:
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
    path = global_config.DATA_RAW_DIR / ticker / filename
    cf = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")

    assert isinstance(cf.index, pd.DatetimeIndex)
    cf.index = cf.index.normalize()

    if not cf.index.is_monotonic_increasing:
        cf = cf.sort_index(ascending=True)

    return cf


def load_ctd_switch_dates(
    ticker: global_config.FutTicker, filename: str = "ctd_switch.csv"
) -> pd.DatetimeIndex:
    """
    Load ctd bond switch dates \n
    Such dates are the first in which a new bond is considered the cheapest

    ## Args:
    * ticker: FutTicker
    * filename: csv filename

    ## Returns:
    * dates: DatetimeIndex
    """
    path = global_config.DATA_RAW_DIR / ticker / filename
    dates = pd.read_csv(path, parse_dates=["CTD Switch Date"])["CTD Switch Date"]
    return pd.DatetimeIndex(dates.dropna().dt.normalize().unique()).sort_values()


def load_fut_rollover_dates(
    ticker: global_config.FutTicker, filename: str = "fut_metadata.csv"
) -> pd.DatetimeIndex:
    """
    Load future rollover dates \n
    Such dates are the last in which a certain future contract is traded

    ## Args:
    * ticker: FutTicker
    * filename: csv filename

    ## Returns:
    * dates: DatetimeIndex
    """
    path = global_config.DATA_RAW_DIR / ticker / filename
    dates = pd.read_csv(path, parse_dates=["Last Trading Date"])["Last Trading Date"]
    return pd.DatetimeIndex(dates.dropna().dt.normalize().unique()).sort_values()


def load_fut_delivery_dates(
    ticker: global_config.FutTicker, filename: str = "fut_metadata.csv"
) -> pd.DatetimeIndex:
    """
    Load future delivery dates \n
    Such dates are the ones in which the short part deliver the underlying to
    the long one, usually a couple of day after the respective rollover date

    ## Args:
    * ticker: FutTicker
    * filename: csv filename

    ## Returns:
    * dates: DatetimeIndex
    """
    path = global_config.DATA_RAW_DIR / ticker / filename
    dates = pd.read_csv(path, parse_dates=["Delivery Date"])["Delivery Date"]
    return pd.DatetimeIndex(dates.dropna().dt.normalize().unique()).sort_values()
