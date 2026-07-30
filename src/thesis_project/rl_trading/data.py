from dataclasses import dataclass
from pathlib import Path

from thesis_project import config as global_config

FBTP_BASE_PATH = global_config.INT_DIR / "fbtp" / "data-microstructure"
FBTS_BASE_PATH = global_config.INT_DIR / "fbts" / "data-microstructure"
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


BASE_FEATURES: dict[str, BaseFeature] = {
    # Cross asset
    "basis": BaseFeature(
        base_id="basis",
        filename=f"gross_basis_{SUFFIX}",
        directory=FBTP_BASE_PATH / "basis",
    ),
    "irr": BaseFeature(
        base_id="irr",
        filename=f"irr_{SUFFIX}",
        directory=FBTP_BASE_PATH / "irr",
    ),
    # Mid prices
    "fut_mid_price": BaseFeature(
        base_id="fut_mid_price",
        filename=f"fut_mid_price_{SUFFIX}",
        directory=FBTP_BASE_PATH / "mid-price",
    ),
    "ctd_mid_price": BaseFeature(
        base_id="ctd_mid_price",
        filename=f"ctd_mid_price_{SUFFIX}",
        directory=FBTP_BASE_PATH / "mid-price",
    ),
    # Micro prices
    "fut_micro_price": BaseFeature(
        base_id="fut_micro_price",
        filename=f"fut_micro_price_{SUFFIX}",
        directory=FBTP_BASE_PATH / "micro-price",
    ),
    "ctd_micro_price": BaseFeature(
        base_id="ctd_micro_price",
        filename=f"ctd_micro_price_{SUFFIX}",
        directory=FBTP_BASE_PATH / "micro-price",
    ),
    # Spreads
    "fut_spread": BaseFeature(
        base_id="fut_spread",
        filename=f"fut_spread_{SUFFIX}",
        directory=FBTP_BASE_PATH / "spread",
    ),
    "ctd_spread": BaseFeature(
        base_id="ctd_spread",
        filename=f"ctd_spread_{SUFFIX}",
        directory=FBTP_BASE_PATH / "spread",
    ),
    # Order Book Imbalance
    "fut_obi_lvl_1": BaseFeature(
        base_id="fut_obi_lvl_1",
        filename=f"fut_obi_lvl_1_scaled_{SUFFIX}",
        directory=FBTP_BASE_PATH / "obi",
    ),
    "ctd_obi_lvl_1": BaseFeature(
        base_id="ctd_obi_lvl_1",
        filename=f"ctd_obi_lvl_1_scaled_{SUFFIX}",
        directory=FBTP_BASE_PATH / "obi",
    ),
    "fut_obi_lvl_2": BaseFeature(
        base_id="fut_obi_lvl_2",
        filename=f"fut_obi_lvl_2_scaled_{SUFFIX}",
        directory=FBTP_BASE_PATH / "obi",
    ),
    "ctd_obi_lvl_2": BaseFeature(
        base_id="ctd_obi_lvl_2",
        filename=f"ctd_obi_lvl_2_scaled_{SUFFIX}",
        directory=FBTP_BASE_PATH / "obi",
    ),
    "fut_obi_lvl_3": BaseFeature(
        base_id="fut_obi_lvl_3",
        filename=f"fut_obi_lvl_3_scaled_{SUFFIX}",
        directory=FBTP_BASE_PATH / "obi",
    ),
    "ctd_obi_lvl_3": BaseFeature(
        base_id="ctd_obi_lvl_3",
        filename=f"ctd_obi_lvl_3_scaled_{SUFFIX}",
        directory=FBTP_BASE_PATH / "obi",
    ),
    # Bid Order Flow
    "fut_bof_lvl_1": BaseFeature(
        base_id="fut_bof_lvl_1",
        filename=f"fut_bof_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "ctd_bof_lvl_1": BaseFeature(
        base_id="ctd_bof_lvl_1",
        filename=f"ctd_bof_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "fut_bof_lvl_2": BaseFeature(
        base_id="fut_bof_lvl_2",
        filename=f"fut_bof_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "ctd_bof_lvl_2": BaseFeature(
        base_id="ctd_bof_lvl_2",
        filename=f"ctd_bof_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "fut_bof_lvl_3": BaseFeature(
        base_id="fut_bof_lvl_3",
        filename=f"fut_bof_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "ctd_bof_lvl_3": BaseFeature(
        base_id="ctd_bof_lvl_3",
        filename=f"ctd_bof_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    # Ask Order Flow
    "fut_aof_lvl_1": BaseFeature(
        base_id="fut_aof_lvl_1",
        filename=f"fut_aof_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "ctd_aof_lvl_1": BaseFeature(
        base_id="ctd_aof_lvl_1",
        filename=f"ctd_aof_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "fut_aof_lvl_2": BaseFeature(
        base_id="fut_aof_lvl_2",
        filename=f"fut_aof_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "ctd_aof_lvl_2": BaseFeature(
        base_id="ctd_aof_lvl_2",
        filename=f"ctd_aof_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "fut_aof_lvl_3": BaseFeature(
        base_id="fut_aof_lvl_3",
        filename=f"fut_aof_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "ctd_aof_lvl_3": BaseFeature(
        base_id="ctd_aof_lvl_3",
        filename=f"ctd_aof_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    # Order Flow Imbalance
    "fut_ofi_lvl_1": BaseFeature(
        base_id="fut_ofi_lvl_1",
        filename=f"fut_ofi_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "ctd_ofi_lvl_1": BaseFeature(
        base_id="ctd_ofi_lvl_1",
        filename=f"ctd_ofi_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "fut_ofi_lvl_2": BaseFeature(
        base_id="fut_ofi_lvl_2",
        filename=f"fut_ofi_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "ctd_ofi_lvl_2": BaseFeature(
        base_id="ctd_ofi_lvl_2",
        filename=f"ctd_ofi_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "fut_ofi_lvl_3": BaseFeature(
        base_id="fut_ofi_lvl_3",
        filename=f"fut_ofi_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    "ctd_ofi_lvl_3": BaseFeature(
        base_id="ctd_ofi_lvl_3",
        filename=f"ctd_ofi_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "ofi",
    ),
    # Slope v1
    "fut_ask_slope_v1_lvl_1": BaseFeature(
        base_id="fut_ask_slope_v1_lvl_1",
        filename=f"fut_ask_slope_v1_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_ask_slope_v1_lvl_1": BaseFeature(
        base_id="ctd_ask_slope_v1_lvl_1",
        filename=f"ctd_ask_slope_v1_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "fut_bid_slope_v1_lvl_1": BaseFeature(
        base_id="fut_bid_slope_v1_lvl_1",
        filename=f"fut_bid_slope_v1_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_bid_slope_v1_lvl_1": BaseFeature(
        base_id="ctd_bid_slope_v1_lvl_1",
        filename=f"ctd_bid_slope_v1_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "fut_ask_slope_v1_lvl_2": BaseFeature(
        base_id="fut_ask_slope_v1_lvl_2",
        filename=f"fut_ask_slope_v1_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_ask_slope_v1_lvl_2": BaseFeature(
        base_id="ctd_ask_slope_v1_lvl_2",
        filename=f"ctd_ask_slope_v1_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "fut_bid_slope_v1_lvl_2": BaseFeature(
        base_id="fut_bid_slope_v1_lvl_2",
        filename=f"fut_bid_slope_v1_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_bid_slope_v1_lvl_2": BaseFeature(
        base_id="ctd_bid_slope_v1_lvl_2",
        filename=f"ctd_bid_slope_v1_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "fut_ask_slope_v1_lvl_3": BaseFeature(
        base_id="fut_ask_slope_v1_lvl_3",
        filename=f"fut_ask_slope_v1_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_ask_slope_v1_lvl_3": BaseFeature(
        base_id="ctd_ask_slope_v1_lvl_3",
        filename=f"ctd_ask_slope_v1_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "fut_bid_slope_v1_lvl_3": BaseFeature(
        base_id="fut_bid_slope_v1_lvl_3",
        filename=f"fut_bid_slope_v1_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_bid_slope_v1_lvl_3": BaseFeature(
        base_id="ctd_bid_slope_v1_lvl_3",
        filename=f"ctd_bid_slope_v1_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    # Slope v2
    "fut_ask_slope_v2_lvl_1": BaseFeature(
        base_id="fut_ask_slope_v2_lvl_1",
        filename=f"fut_ask_slope_v2_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_ask_slope_v2_lvl_1": BaseFeature(
        base_id="ctd_ask_slope_v2_lvl_1",
        filename=f"ctd_ask_slope_v2_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "fut_bid_slope_v2_lvl_1": BaseFeature(
        base_id="fut_bid_slope_v2_lvl_1",
        filename=f"fut_bid_slope_v2_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_bid_slope_v2_lvl_1": BaseFeature(
        base_id="ctd_bid_slope_v2_lvl_1",
        filename=f"ctd_bid_slope_v2_lvl_1_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "fut_ask_slope_v2_lvl_2": BaseFeature(
        base_id="fut_ask_slope_v2_lvl_2",
        filename=f"fut_ask_slope_v2_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_ask_slope_v2_lvl_2": BaseFeature(
        base_id="ctd_ask_slope_v2_lvl_2",
        filename=f"ctd_ask_slope_v2_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "fut_bid_slope_v2_lvl_2": BaseFeature(
        base_id="fut_bid_slope_v2_lvl_2",
        filename=f"fut_bid_slope_v2_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_bid_slope_v2_lvl_2": BaseFeature(
        base_id="ctd_bid_slope_v2_lvl_2",
        filename=f"ctd_bid_slope_v2_lvl_2_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "fut_ask_slope_v2_lvl_3": BaseFeature(
        base_id="fut_ask_slope_v2_lvl_3",
        filename=f"fut_ask_slope_v2_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_ask_slope_v2_lvl_3": BaseFeature(
        base_id="ctd_ask_slope_v2_lvl_3",
        filename=f"ctd_ask_slope_v2_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "fut_bid_slope_v2_lvl_3": BaseFeature(
        base_id="fut_bid_slope_v2_lvl_3",
        filename=f"fut_bid_slope_v2_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
    "ctd_bid_slope_v2_lvl_3": BaseFeature(
        base_id="ctd_bid_slope_v2_lvl_3",
        filename=f"ctd_bid_slope_v2_lvl_3_{SUFFIX}",
        directory=FBTP_BASE_PATH / "slope",
    ),
}
