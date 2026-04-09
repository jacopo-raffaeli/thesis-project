import re
from datetime import time
from pathlib import Path

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
BID_PRICE_COL_PATTERN = re.compile(r"^L\d+-BidPrice$")
ASK_PRICE_COL_PATTERN = re.compile(r"^L\d+-AskPrice$")
BID_SIZE_COL_PATTERN = re.compile(r"^L\d+-BidSize$")
ASK_SIZE_COL_PATTERN = re.compile(r"^L\d+-AskSize$")
FUT_COL_TO_DROP = ["#RIC", "Domain", "GMT Offset", "Type"]
