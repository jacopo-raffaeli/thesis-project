"""Data file paths configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from thesis_project import config as global_config


@dataclass
class DataPathsConfig:
    """Data file paths for a given ticker."""

    ticker: str
    target: Tuple[str, Path]
    features: Dict[str, Path]


def get_data_paths(ticker: str) -> DataPathsConfig:
    """Get data file paths for a given ticker."""

    ms_path = global_config.INT_DIR / ticker / "data-microstructure"

    # Target data path
    target = (
        "basis",
        ms_path / "basis" / "gross_basis_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "irr", ms_path / "irr" / "irr_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
    )

    # Feature data paths
    # fmt: off
    features = {
        # Cross-asset
        "basis": ms_path / "basis" / "gross_basis_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "irr": ms_path / "irr" / "irr_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Mid Prices
        # "fut_mid_price": ms_path / "mid-price" / "fut_mid_price_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_mid_price": ms_path / "mid-price" / "ctd_mid_price_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Micro Prices
        "fut_micro_price": ms_path / "micro-price" / "fut_micro_price_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "ctd_micro_price": ms_path / "micro-price" / "ctd_micro_price_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Spreads
        "fut_spread": ms_path / "spread" / "fut_spread_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "ctd_spread": ms_path / "spread" / "ctd_spread_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Order Book Imbalance
        "fut_obi_lvl_1": ms_path / "obi" / "fut_obi_lvl_1_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "ctd_obi_lvl_1": ms_path / "obi" / "ctd_obi_lvl_1_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "fut_obi_lvl_2": ms_path / "obi" / "fut_obi_lvl_2_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "ctd_obi_lvl_2": ms_path / "obi" / "ctd_obi_lvl_2_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "fut_obi_lvl_3": ms_path / "obi" / "fut_obi_lvl_3_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "ctd_obi_lvl_3": ms_path / "obi" / "ctd_obi_lvl_3_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Bid Order Flow
        # "fut_bof_lvl_1": ms_path / "ofi" / "fut_bof_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bof_lvl_1": ms_path / "ofi" / "ctd_bof_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_bof_lvl_2": ms_path / "ofi" / "fut_bof_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bof_lvl_2": ms_path / "ofi" / "ctd_bof_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_bof_lvl_3": ms_path / "ofi" / "fut_bof_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bof_lvl_3": ms_path / "ofi" / "ctd_bof_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Ask Order Flow
        # "fut_aof_lvl_1": ms_path / "ofi" / "fut_aof_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_aof_lvl_1": ms_path / "ofi" / "ctd_aof_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_aof_lvl_2": ms_path / "ofi" / "fut_aof_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_aof_lvl_2": ms_path / "ofi" / "ctd_aof_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_aof_lvl_3": ms_path / "ofi" / "fut_aof_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_aof_lvl_3": ms_path / "ofi" / "ctd_aof_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Order Flow Imbalance
        "fut_ofi_lvl_1": ms_path / "ofi" / "fut_ofi_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "ctd_ofi_lvl_1": ms_path / "ofi" / "ctd_ofi_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "fut_ofi_lvl_2": ms_path / "ofi" / "fut_ofi_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "ctd_ofi_lvl_2": ms_path / "ofi" / "ctd_ofi_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "fut_ofi_lvl_3": ms_path / "ofi" / "fut_ofi_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "ctd_ofi_lvl_3": ms_path / "ofi" / "ctd_ofi_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Slope v1
        # "fut_ask_slope_v1_lvl_1": ms_path / "slope" / "fut_ask_slope_v1_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_ask_slope_v1_lvl_1": ms_path / "slope" / "ctd_ask_slope_v1_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_bid_slope_v1_lvl_1": ms_path / "slope" / "fut_bid_slope_v1_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bid_slope_v1_lvl_1": ms_path / "slope" / "ctd_bid_slope_v1_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_ask_slope_v1_lvl_2": ms_path / "slope" / "fut_ask_slope_v1_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_ask_slope_v1_lvl_2": ms_path / "slope" / "ctd_ask_slope_v1_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_bid_slope_v1_lvl_2": ms_path / "slope" / "fut_bid_slope_v1_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bid_slope_v1_lvl_2": ms_path / "slope" / "ctd_bid_slope_v1_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_ask_slope_v1_lvl_3": ms_path / "slope" / "fut_ask_slope_v1_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_ask_slope_v1_lvl_3": ms_path / "slope" / "ctd_ask_slope_v1_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_bid_slope_v1_lvl_3": ms_path / "slope" / "fut_bid_slope_v1_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bid_slope_v1_lvl_3": ms_path / "slope" / "ctd_bid_slope_v1_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Slope v2
        # "fut_ask_slope_v2_lvl_1": ms_path / "slope" / "fut_ask_slope_v2_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_ask_slope_v2_lvl_1": ms_path / "slope" / "ctd_ask_slope_v2_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_bid_slope_v2_lvl_1": ms_path / "slope" / "fut_bid_slope_v2_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bid_slope_v2_lvl_1": ms_path / "slope"/ "ctd_bid_slope_v2_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_ask_slope_v2_lvl_2": ms_path / "slope" / "fut_ask_slope_v2_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_ask_slope_v2_lvl_2": ms_path / "slope" / "ctd_ask_slope_v2_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_bid_slope_v2_lvl_2": ms_path / "slope" / "fut_bid_slope_v2_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bid_slope_v2_lvl_2": ms_path / "slope" / "ctd_bid_slope_v2_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_ask_slope_v2_lvl_3": ms_path / "slope" / "fut_ask_slope_v2_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_ask_slope_v2_lvl_3": ms_path / "slope" / "ctd_ask_slope_v2_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_bid_slope_v2_lvl_3": ms_path / "slope" / "fut_bid_slope_v2_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bid_slope_v2_lvl_3": ms_path / "slope" / "ctd_bid_slope_v2_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
    }
    # fmt: on

    return DataPathsConfig(ticker=ticker, target=target, features=features)
