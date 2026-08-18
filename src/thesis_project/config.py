import datetime
import re
import zoneinfo
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Literal

import pandas as pd

# ==============================================================================
# DIRECTORIES
# ==============================================================================

# Project root directory
ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INT_DIR = DATA_DIR / "interim"
DATA_PRO_DIR = DATA_DIR / "processed"

# Results directories
RES_DIR = ROOT / "results"
RES_FIG_DIR = RES_DIR / "figures"
RES_TAB_DIR = RES_DIR / "tables"
RES_EXP_DIR = RES_DIR / "experiments"

# Source directories
SRC_DIR = ROOT / "src"
SRC_THESIS_DIR = SRC_DIR / "thesis_project"


# ==============================================================================
# MARKET CONFIG
# ==============================================================================


@dataclass(frozen=True)
class MarketConfig:
    name: str
    opening_time: datetime.time
    closing_time: datetime.time
    tz: zoneinfo.ZoneInfo


MTS = MarketConfig(
    name="mts",
    opening_time=datetime.time(8, 0, 0),
    closing_time=datetime.time(17, 30, 0),
    tz=zoneinfo.ZoneInfo("Europe/Berlin"),
)

EUREX = MarketConfig(
    name="eurex",
    opening_time=datetime.time(8, 0, 0),
    closing_time=datetime.time(19, 0, 0),
    tz=zoneinfo.ZoneInfo("Europe/Berlin"),
)

MARKETS = {market.name: market for market in [MTS, EUREX]}


# ==============================================================================
# ASSET CONFIG
# ==============================================================================

FutTicker = Literal["fbtp", "fbts"]
AssetSymbol = Literal[FutTicker, "btp"]
AssetFamily = Literal["bond", "future"]
AssetRole = Literal["ctd", "fut"]


@dataclass(frozen=True)
class AssetConfig:
    """
    Dataclass for asset configuration

    ## Args:
    * symbol: asset symbol
    * family: asset family
    * role: useful to distinguish the role of the asset in the basis
    * par_value: The reference value quoted prices refers to in euros
    * min_trade_face_vale: minimum tradable size in euros
    * tick_size_perc: tick size expressed as a percent of the par value
    """

    symbol: AssetSymbol
    family: AssetFamily
    role: AssetRole
    par_value: float
    contract_size: float
    tick_size_perc: float
    market: MarketConfig
    description: str

    @property
    def tick_size_euro(self) -> float:
        return self.contract_size * (self.tick_size_perc / 100)


BTP = AssetConfig(
    symbol="btp",
    family="bond",
    role="ctd",
    par_value=100,
    contract_size=2_000_000,
    tick_size_perc=0.01,
    market=MTS,
    description="Buoni del Tesoro Pluriennali",
)

FBTP = AssetConfig(
    symbol="fbtp",
    family="future",
    role="fut",
    par_value=100,
    contract_size=100_000,
    tick_size_perc=0.01,
    market=EUREX,
    description="Long-Term Euro-BTP Futures",
)

FBTS = AssetConfig(
    symbol="fbts",
    family="future",
    role="fut",
    par_value=100,
    contract_size=100_000,
    tick_size_perc=0.005,
    market=EUREX,
    description="Short-Term Euro-BTP Futures",
)

ASSETS = {asset.symbol: asset for asset in [BTP, FBTP, FBTS]}


# ==============================================================================
# LOB METADATA
# ==============================================================================

LobSide = Literal["ask", "bid"]
LobColumn = Literal["price", "size"]


@dataclass(frozen=True)
class LobMetadata:
    n_levels: int
    freq: str
    index_name: str = "timestamp"
    sides: tuple[LobSide, ...] = ("bid", "ask")
    column_types: tuple[LobColumn, ...] = ("price", "size")
    column_format: str = "L{level}-{side}{column_type}"

    def __post_init__(self):
        if self.n_levels <= 0:
            raise ValueError("LOB levels must be a positive integer")

        if not self.sides:
            raise ValueError("At least one LOB side is required")

        if not self.column_types:
            raise ValueError("At least one column type is required")

    @property
    def levels(self) -> list[int]:
        return list(range(1, self.n_levels + 1))

    @property
    def bid_column(self) -> str:
        return self.column_format.format(side="Bid")

    @property
    def ask_column(self) -> str:
        return self.column_format.format(side="Ask")

    @property
    def price_column(self) -> str:
        return self.column_format.format(column_type="Price")

    @property
    def size_column(self) -> str:
        return self.column_format.format(column_type="Size")

    @property
    def bid_price_column(self) -> str:
        return self.bid_column.format(column_type="Price")

    @property
    def bid_size_column(self) -> str:
        return self.bid_column.format(column_type="Size")

    @property
    def ask_price_column(self) -> str:
        return self.ask_column.format(column_type="Price")

    @property
    def ask_size_column(self) -> str:
        return self.ask_column.format(column_type="Size")

    def column(self, *, level: int, side: LobSide, column_type: LobColumn) -> str:
        return self.column_format.format(
            level=level,
            side=side.capitalize(),
            column_type=column_type.capitalize(),
        )

    def columns(
        self,
        *,
        levels: list[int] | int | None = None,
        sides: tuple[LobSide, ...] | LobSide | None = None,
        column_types: tuple[LobColumn, ...] | LobColumn | None = None,
    ) -> list[str]:
        if isinstance(levels, int):
            levels = [levels]
        elif levels is None:
            levels = self.levels

        if isinstance(sides, str):
            sides = (sides,)
        elif sides is None:
            sides = self.sides

        if isinstance(column_types, str):
            column_types = (column_types,)
        elif column_types is None:
            column_types = self.column_types

        return [
            self.column(level=level, side=side, column_type=column_type)
            for level in sorted(levels)
            for side in sides
            for column_type in column_types
        ]


LOB_METADATA = LobMetadata(
    n_levels=10,
    freq="1s",
    index_name="timestamp",
    sides=("bid", "ask"),
    column_types=("price", "size"),
)


# ==============================================================================

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
VALID_TICKERS = {"fbtp", "fbts"}

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
    DATA_RAW_DIR / "fbtp" / "fut_metadata.csv", parse_dates=["Last Trading Date", "Delivery Date"]
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
