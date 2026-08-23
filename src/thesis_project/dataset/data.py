from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from thesis_project import config

DATA_INT_FBTP_MS_DIR = config.DATA_INT_DIR / "fbtp" / "data-microstructure"
DATA_INT_FBTS_MS_DIR = config.DATA_INT_DIR / "fbts" / "data-microstructure"
SUFFIX = "freq_1s_from_2022_08_01_to_2024_12_30.parquet"


@dataclass(frozen=True)
class BaseFeature:
    base_id: str
    filename: str
    directory: Path
    description: str | None = None

    def __post_init__(self):
        if not self.directory.exists():
            raise ValueError(f"Missing directory: '{self.directory.relative_to(config.ROOT)}'")

        if not self.directory.is_dir():
            raise ValueError(f"Not a directory: '{self.directory.relative_to(config.ROOT)}'")

        if not self.path.exists():
            raise ValueError(f"Missing file '{self.path.relative_to(config.ROOT)}'")

        if not self.path.is_file():
            raise ValueError(f"Not a file '{self.path.relative_to(config.ROOT)}'")

    @property
    def path(self) -> Path:
        return self.directory / self.filename


# fmt: off
BASE_FEATURES: dict[str, BaseFeature] = {
    # Cross role
    "basis": BaseFeature(
        base_id="basis",
        filename=f"gross_basis_{SUFFIX}",
        directory=DATA_INT_FBTP_MS_DIR / "basis",
    ),
    "irr": BaseFeature(
        base_id="irr",
        filename=f"irr_{SUFFIX}",
        directory=DATA_INT_FBTP_MS_DIR / "irr",
    ),

    # Mid prices
    **{
        f"{role}_mid_price": BaseFeature(
            base_id=f"{role}_mid_price",
            filename=f"{role}_mid_price_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "mid-price",
        )
        for role in get_args(config.AssetRole)
    },

    # Micro prices
    **{
        f"{role}_micro_price": BaseFeature(
            base_id=f"{role}_micro_price",
            filename=f"{role}_micro_price_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "micro-price",
        )
        for role in get_args(config.AssetRole)
    },

    # Spreads
    **{
        f"{role}_spread": BaseFeature(
            base_id=f"{role}_spread",
            filename=f"{role}_spread_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "spread",
        )
        for role in get_args(config.AssetRole)
    },

    # Order Book Imbalance
    **{
        f"{role}_obi_lvl_{level}": BaseFeature(
            base_id=f"{role}_obi_lvl_{level}",
            filename=f"{role}_obi_lvl_{level}_scaled_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "obi",
        )
        for role in get_args(config.AssetRole)
        for level in range(1, 4)
    },

    # Bid/Ask Order Flow
    **{
        f"{role}_{side}of_lvl_{level}": BaseFeature(
            base_id=f"{role}_{side}of_lvl_{level}",
            filename=f"{role}_{side}of_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "ofi",
        )
        for role in get_args(config.AssetRole)
        for side in ("b", "a")
        for level in range(1, 4)
    },

    # Order Flow Imbalance
    **{
        f"{role}_ofi_lvl_{level}": BaseFeature(
            base_id=f"{role}_ofi_lvl_{level}",
            filename=f"{role}_ofi_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "ofi",
        )
        for role in get_args(config.AssetRole)
        for level in range(1, 4)
    },

    # Slope v1
    **{
        f"{role}_{side}_slope_v1_lvl_{level}": BaseFeature(
            base_id=f"{role}_{side}_slope_v1_lvl_{level}",
            filename=f"{role}_{side}_slope_v1_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "slope",
        )
        for role in get_args(config.AssetRole)
        for side in ("ask", "bid")
        for level in range(1, 4)
    },

    # Slope v2
    **{
        f"{role}_{side}_slope_v2_lvl_{level}": BaseFeature(
            base_id=f"{role}_{side}_slope_v2_lvl_{level}",
            filename=f"{role}_{side}_slope_v2_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "slope",
        )
        for role in get_args(config.AssetRole)
        for side in ("ask", "bid")
        for level in range(1, 4)
    },

    # Price levels
    **{
        f"{role}_{side}_price_lvl_{level}": BaseFeature(
            base_id=f"{role}_{side}_price_lvl_{level}",
            filename=f"{role}_{side}_price_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "price",
        )
        for role in get_args(config.AssetRole)
        for side in ("bid", "ask")
        for level in range(1, config.LOB.n_levels + 1)
    },

    # Size levels
    **{
        f"{role}_{side}_size_lvl_{level}": BaseFeature(
            base_id=f"{role}_{side}_size_lvl_{level}",
            filename=f"{role}_{side}_size_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "size",
        )
        for role in get_args(config.AssetRole)
        for side in ("bid", "ask")
        for level in range(1, config.LOB.n_levels + 1)
    },
}
# fmt: on
