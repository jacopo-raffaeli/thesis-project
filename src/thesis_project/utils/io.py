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
