import re
from datetime import time
from pathlib import Path

import pandas as pd

# DIRECTORIES

# Project root directory
ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INT_DIR = DATA_DIR / "interim"
PRO_DIR = DATA_DIR / "processed"

# Results directories
RES_DIR = ROOT / "results"
FIG_DIR = RES_DIR / "figures"
TAB_DIR = RES_DIR / "tables"

# Sources directories
SRC_DIR = ROOT / "src"
SRC_THESIS_DIR = SRC_DIR / "thesis_project"

# Plot settings directories
PLT_DIR = SRC_THESIS_DIR / "plotting"
MPL_CONFIG = PLT_DIR / "config.mplstyle"

# TIME SETTINGS

# Market hours settings
FUT_MRKT_OPEN = time(8, 0)
FUT_MRKT_CLOSE = time(19, 0)
CTD_MRKT_OPEN = time(8, 0)
CTD_MRKT_CLOSE = time(17, 30)

# Common time settings for joint analysis
START_TIME = time(9, 0)
END_TIME = time(17, 0)

# Year range
FBTP_START_YEAR = 2022
FBTP_END_YEAR = 2024
FBTS_START_YEAR = 2022
FBTS_END_YEAR = 2025

# LOB STRUCTURE

LEVELS = 10

DATE_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})")
DATE_TEMPLATE = "{year:04d}_{month:02d}_{day:02d}"

BID_PRICE_COL_PATTERN = re.compile(r"^L\d+-BidPrice$")
ASK_PRICE_COL_PATTERN = re.compile(r"^L\d+-AskPrice$")
BID_SIZE_COL_PATTERN = re.compile(r"^L\d+-BidSize$")
ASK_SIZE_COL_PATTERN = re.compile(r"^L\d+-AskSize$")
BID_PRICE_COL_TEMPLATE = "L{lvl}-BidPrice"
ASK_PRICE_COL_TEMPLATE = "L{lvl}-AskPrice"
BID_SIZE_COL_TEMPLATE = "L{lvl}-BidSize"
ASK_SIZE_COL_TEMPLATE = "L{lvl}-AskSize"

FUT_COL_TO_DROP = ["#RIC", "Domain", "GMT Offset", "Type"]

SERIES_FILENAME_TEMPLATE = "{series_name}_freq_{freq}_from_{start_date}_to_{end_date}.{ext}"

# TICK SIZE

FBTP_TICK_SIZE_PERC = 0.01
FBTP_TICK_SIZE_EURO = 10

FBTS_TICK_SIZE_PERC = 0.005
FBTS_TICK_SIZE_EURO = 5

CTD_TICK_SIZE_PERC = 0.01
CTD_TICK_SIZE_EURO = 0.01

# DATES TO EXCLUDE

fbts_fut_meta_df = pd.read_csv(
    RAW_DIR / "fbtp" / "fut_metadata.csv", parse_dates=["Last Trading Date", "Delivery Date"]
)
# fbts_fut_meta_df = pd.read_csv(
#     RAW_DIR / "fbts" / "fut_metadata.csv", parse_dates=["Last Trading Date", "Delivery Date"]
# )

DATES_TO_EXCLUDE = {
    "fbtp": {
        "structural": [
            "2022-08-29",  # FUT LOB stops early
            "2024-12-02",  # FUT LOB spread violation (1)
            "2024-06-11",  # CTD LOB spread violation (127)
            "2023-09-06",  # FUT LOB price order violation (2)
            "2023-12-06",  # FUT LOB price order violation (9)
        ],
        "fut_last_trading_days": fbts_fut_meta_df["Last Trading Date"].dt.date.unique().tolist(),
    },
    "fbts": {"structural": [], "last_trading_days": []},
}
