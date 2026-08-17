from dataclasses import dataclass
from pathlib import Path

from thesis_project import config as global_config

DATA_INT_FBTP_MS_DIR = global_config.DATA_INT_DIR / "fbtp" / "data-microstructure"
DATA_INT_FBTS_MS_DIR = global_config.DATA_INT_DIR / "fbts" / "data-microstructure"
SUFFIX = "freq_1s_from_2022_08_01_to_2024_12_30.parquet"


@dataclass(frozen=True)
class BaseFeature:
    base_id: str
    filename: str
    directory: Path
    description: str | None = None

    def __post_init__(self):
        if not self.directory.exists():
            raise ValueError(
                f"Missing directory: '{self.directory.relative_to(global_config.ROOT)}'"
            )

        if not self.directory.is_dir():
            raise ValueError(f"Not a directory: '{self.directory.relative_to(global_config.ROOT)}'")

        if not self.path.exists():
            raise ValueError(f"Missing file '{self.path.relative_to(global_config.ROOT)}'")

        if not self.path.is_file():
            raise ValueError(f"Not a file '{self.path.relative_to(global_config.ROOT)}'")

    @property
    def path(self) -> Path:
        return self.directory / self.filename


# fmt: off
BASE_FEATURES: dict[str, BaseFeature] = {
    # Cross asset
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
        f"{asset}_mid_price": BaseFeature(
            base_id=f"{asset}_mid_price",
            filename=f"{asset}_mid_price_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "mid-price",
        )
        for asset in ("fut", "ctd")
    },

    # Micro prices
    **{
        f"{asset}_micro_price": BaseFeature(
            base_id=f"{asset}_micro_price",
            filename=f"{asset}_micro_price_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "micro-price",
        )
        for asset in ("fut", "ctd")
    },

    # Spreads
    **{
        f"{asset}_spread": BaseFeature(
            base_id=f"{asset}_spread",
            filename=f"{asset}_spread_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "spread",
        )
        for asset in ("fut", "ctd")
    },

    # Order Book Imbalance
    **{
        f"{asset}_obi_lvl_{level}": BaseFeature(
            base_id=f"{asset}_obi_lvl_{level}",
            filename=f"{asset}_obi_lvl_{level}_scaled_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "obi",
        )
        for asset in ("fut", "ctd")
        for level in range(1, 4)
    },

    # Bid/Ask Order Flow
    **{
        f"{asset}_{side}of_lvl_{level}": BaseFeature(
            base_id=f"{asset}_{side}of_lvl_{level}",
            filename=f"{asset}_{side}of_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "ofi",
        )
        for asset in ("fut", "ctd")
        for side in ("b", "a")
        for level in range(1, 4)
    },

    # Order Flow Imbalance
    **{
        f"{asset}_ofi_lvl_{level}": BaseFeature(
            base_id=f"{asset}_ofi_lvl_{level}",
            filename=f"{asset}_ofi_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "ofi",
        )
        for asset in ("fut", "ctd")
        for level in range(1, 4)
    },

    # Slope v1
    **{
        f"{asset}_{side}_slope_v1_lvl_{level}": BaseFeature(
            base_id=f"{asset}_{side}_slope_v1_lvl_{level}",
            filename=f"{asset}_{side}_slope_v1_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "slope",
        )
        for asset in ("fut", "ctd")
        for side in ("ask", "bid")
        for level in range(1, 4)
    },

    # Slope v2
    **{
        f"{asset}_{side}_slope_v2_lvl_{level}": BaseFeature(
            base_id=f"{asset}_{side}_slope_v2_lvl_{level}",
            filename=f"{asset}_{side}_slope_v2_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "slope",
        )
        for asset in ("fut", "ctd")
        for side in ("ask", "bid")
        for level in range(1, 4)
    },

    # Price levels
    **{
        f"{asset}_{side}_price_lvl_{level}": BaseFeature(
            base_id=f"{asset}_{side}_price_lvl_{level}",
            filename=f"{asset}_{side}_price_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "price",
        )
        for asset in ("fut", "ctd")
        for side in ("bid", "ask")
        for level in range(1, 11)
    },

    # Size levels
    **{
        f"{asset}_{side}_size_lvl_{level}": BaseFeature(
            base_id=f"{asset}_{side}_size_lvl_{level}",
            filename=f"{asset}_{side}_size_lvl_{level}_{SUFFIX}",
            directory=DATA_INT_FBTP_MS_DIR / "size",
        )
        for asset in ("fut", "ctd")
        for side in ("bid", "ask")
        for level in range(1, 11)
    },
}
# fmt: on
