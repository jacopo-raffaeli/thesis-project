import datetime
import re

import dataframe_image as dfi
import pandas as pd

from thesis_project import config as global_config


def df_to_png(df, path, filename):
    dfi.export(df, path / f"{filename}.png", table_conversion="chrome")


def load_cf(ticker: str, filename: str = "daily_cf.csv") -> pd.DataFrame:
    path = global_config.DATA_RAW_DIR / ticker
    cf = pd.read_csv(path / filename, parse_dates=["Date"]).set_index("Date")

    if not cf.index.is_monotonic_increasing:
        cf = cf.sort_index()

    return cf


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
