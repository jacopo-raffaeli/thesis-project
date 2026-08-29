from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from thesis_project import config
from thesis_project.dataset import microstructure

FBTP_DIR = config.DATA_PRO_DIR / "fbtp" / "microstructure"
FBTS_DIR = config.DATA_PRO_DIR / "fbts" / "microstructure"


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
        return (self.directory / self.filename).with_suffix(".parquet")


# fmt: off
BASE_FEATURES: dict[str, BaseFeature] = {
    # Price levels
    **{
        f"{role}_{side}_price_{level}": BaseFeature(
            base_id=f"{role}_{side}_price_{level}",
            filename=f"{role}_{side}_price_{level}",
            directory=FBTP_DIR / "price",
        )
        for role in get_args(config.AssetRole)
        for side in get_args(config.LobSide)
        for level in range(1, config.LOB.n_levels + 1)
    },

    # Size levels
    **{
        f"{role}_{side}_size_{level}": BaseFeature(
            base_id=f"{role}_{side}_size_{level}",
            filename=f"{role}_{side}_size_{level}",
            directory=FBTP_DIR / "size",
        )
        for role in get_args(config.AssetRole)
        for side in get_args(config.LobSide)
        for level in range(1, config.LOB.n_levels + 1)
    },

    # Mid prices
    **{
        f"{role}_mid_price": BaseFeature(
            base_id=f"{role}_mid_price",
            filename=f"{role}_mid_price",
            directory=FBTP_DIR / "mid-price",
        )
        for role in get_args(config.AssetRole)
    },

    # Micro prices
    **{
        f"{role}_micro_price": BaseFeature(
            base_id=f"{role}_micro_price",
            filename=f"{role}_micro_price",
            directory=FBTP_DIR / "micro-price",
        )
        for role in get_args(config.AssetRole)
    },

    # Spreads
    **{
        f"{role}_spread": BaseFeature(
            base_id=f"{role}_spread",
            filename=f"{role}_spread",
            directory=FBTP_DIR / "spread",
        )
        for role in get_args(config.AssetRole)
    },

    # Order Book Imbalance
    **{
        f"{role}_obi_{max_level}": BaseFeature(
            base_id=f"{role}_obi_{max_level}",
            filename=f"{role}_obi_{max_level}",
            directory=FBTP_DIR / "obi",
        )
        for role in get_args(config.AssetRole)
        for max_level in [1, 2, 3]
    },

    # Bid/Ask Order Flow
    **{
        f"{role}_{str(side)[0]}of_{level}": BaseFeature(
            base_id=f"{role}_{side}of_{level}",
            filename=f"{role}_{side}of_{level}",
            directory=FBTP_DIR / "ofi",
        )
        for role in get_args(config.AssetRole)
        for side in get_args(config.LobSide)
        for level in [1, 2, 3]
    },

    # Order Flow Imbalance
    **{
        f"{role}_ofi_{level}": BaseFeature(
            base_id=f"{role}_ofi_{level}",
            filename=f"{role}_ofi_{level}",
            directory=FBTP_DIR / "ofi",
        )
        for role in get_args(config.AssetRole)
        for level in [1, 2, 3]
    },

    # Slope
    **{
        f"{role}_{side}_slope__{max_level}_{microstructure.SLOPE_DICT[slope_type]}": BaseFeature(
            base_id=f"{role}_{side}_slope_{microstructure.SLOPE_DICT[slope_type]}_{max_level}",
            filename=f"{role}_{side}_slope_{microstructure.SLOPE_DICT[slope_type]}_{max_level}",
            directory=FBTP_DIR / "slope",
        )
        for role in get_args(config.AssetRole)
        for side in get_args(config.LobSide)
        for max_level in [1, 2, 3]
        for slope_type in get_args(microstructure.SlopeType)
    },

    # Cross asset
    "basis": BaseFeature(
        base_id="basis",
        filename="basis",
        directory=FBTP_DIR / "basis",
    ),
    "irr": BaseFeature(
        base_id="irr",
        filename="irr",
        directory=FBTP_DIR / "irr",
    ),
}
# fmt: on
