import datetime
import re
import zoneinfo
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import ClassVar, Literal, get_args

import matplotlib.pyplot as plt

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
# GENERAL
# ==============================================================================

# Standard financial naming conventions
FutTicker = Literal["fbtp", "fbts"]
AssetSymbol = Literal[FutTicker, "btp"]
AssetFamily = Literal["bond", "future"]
AssetRole = Literal["ctd", "fut"]

# Standard LOB naming conventions
LobSide = Literal["ask", "bid"]
LobColumn = Literal["price", "size"]

# Standard
IncludeCost = Literal[
    "On",
    "Off",
]

IncludeVolume = Literal[
    "On",
    "Off",
]

RoundContract = Literal[
    "On",
    "Off",
]

# Standard timeS for fut-ctd joint analysis
DEFAULT_OPENING_TIME = time(9, 0, 0)
DEFAULT_CLOSING_TIME = time(17, 0, 0)


# ==============================================================================
# MARKET CONFIG
# ==============================================================================


@dataclass(frozen=True)
class MarketConfig:
    """
    Dataclass for market configuration

    * name: market name
    * opening_time: opening trading time
    * closing_time: closing trading time
    * tz: market timezone
    """

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
    price_tick_perc: float
    size_tick: float
    market: MarketConfig
    description: str

    @property
    def price_tick_euro(self) -> float:
        return self.contract_size * (self.price_tick_perc / 100)


BTP = AssetConfig(
    symbol="btp",
    family="bond",
    role="ctd",
    par_value=100,
    contract_size=2_000_000,
    price_tick_perc=0.01,
    size_tick=0.5,
    market=MTS,
    description="Buoni del Tesoro Pluriennali",
)

FBTP = AssetConfig(
    symbol="fbtp",
    family="future",
    role="fut",
    par_value=100,
    contract_size=100_000,
    price_tick_perc=0.01,
    size_tick=1.0,
    market=EUREX,
    description="Long-Term Euro-BTP Futures",
)

FBTS = AssetConfig(
    symbol="fbts",
    family="future",
    role="fut",
    par_value=100,
    contract_size=100_000,
    price_tick_perc=0.005,
    size_tick=1.0,
    market=EUREX,
    description="Short-Term Euro-BTP Futures",
)

ASSETS = {asset.symbol: asset for asset in [BTP, FBTP, FBTS]}

ASSET_BY_TICKER_ROLE: dict[tuple[FutTicker, AssetRole], AssetConfig] = {
    ("fbtp", "ctd"): BTP,
    ("fbtp", "fut"): FBTP,
    ("fbts", "ctd"): BTP,
    ("fbts", "fut"): FBTS,
}


# ==============================================================================
# LOB METADATA
# ==============================================================================


@dataclass(frozen=True)
class LobConfig:
    """
    Dataclass for LOB metadata

    ## Args:
    * n_levels: LOB levels
    * index_name: name of the index column
    * sides: ids of the sides of a LOB level
    * column_types: ids of the column_types of a LOB level
    * column_format: formattable string for the canonical column name
    """

    ROLES: ClassVar[tuple[AssetRole, ...]] = get_args(AssetRole)
    FILENAME: ClassVar[str] = "{role}_lob_freq_{freq}_{yyyy}_{mm}_{dd}.parquet"

    n_levels: int
    freq: str
    column_format: str = "L{level}-{side}{column_type}"
    index_name: str = "timestamp"
    columns: tuple[LobColumn, ...] = ("price", "size")
    sides: tuple[LobSide, ...] = ("bid", "ask")

    def __post_init__(self):
        if self.n_levels <= 0:
            raise ValueError("LOB levels must be a positive integer")

        if not self.sides:
            raise ValueError("At least one LOB side is required")

        if not self.columns:
            raise ValueError("At least one column type is required")

    def filename(self, *, role: AssetRole, date: datetime.date) -> str:
        if role not in self.ROLES:
            raise ValueError(f"Invalid LOB {role=!r}")

        return self.FILENAME.format(
            role=role,
            freq=self.freq,
            yyyy=f"{date.year:04d}",
            mm=f"{date.month:02d}",
            dd=f"{date.day:02d}",
        )

    @property
    def filename_re(self) -> re.Pattern[str]:
        return re.compile(
            rf"(?P<role>{'|'.join(map(re.escape, self.ROLES))})"
            rf"_lob_freq_{re.escape(self.freq)}"
            rf"_(?P<year>\d{{4}})"
            rf"_(?P<month>\d{{2}})"
            rf"_(?P<day>\d{{2}})"
            rf"\.parquet"
        )

    @property
    def levels(self) -> list[int]:
        return list(range(1, self.n_levels + 1))

    def get_column(self, *, level: int, side: LobSide, column_type: LobColumn) -> str:
        return self.column_format.format(
            level=level,
            side=side.capitalize(),
            column_type=column_type.capitalize(),
        )

    def get_columns(
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
            column_types = self.columns

        return [
            self.get_column(level=level, side=side, column_type=column_type)
            for level in sorted(levels)
            for side in sides
            for column_type in column_types
        ]


LOB = LobConfig(
    n_levels=10,
    freq="1s",
    index_name="timestamp",
    sides=("bid", "ask"),
    columns=("price", "size"),
)

# ==============================================================================
# DATES TO EXCLUDE
# ==============================================================================

DateType = Literal[
    "Critical",
    "Rollover",
    "Extra",
]

DEFAULT_EXCLUDED_DATES: dict[DateType, tuple[int, int] | None] = {
    "Critical": None,
    "Rollover": (2, 0),
    "Extra": None,
}


# ==============================================================================
# PLOTTING
# ==============================================================================


# Matplotlib.pyplot global settings
def default_plt():
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-paper")


# Apply default settings
def default():
    default_plt()
