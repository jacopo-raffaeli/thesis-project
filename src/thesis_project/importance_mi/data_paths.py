"""Data file paths configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from thesis_project import config as global_config


@dataclass
class DataPathsConfig:
    """Data file paths for a given ticker."""

    ticker: str
    targets: Dict[str, Path]
    features: Dict[str, Path]


def get_data_paths(ticker: str) -> DataPathsConfig:
    """Get data file paths for a given ticker."""
    if ticker not in ["fbtp", "fbts"]:
        raise ValueError(f"Invalid ticker: {ticker}. Must be 'fbtp' or 'fbts'")

    ms_path = global_config.DATA_INT_DIR / ticker / "data-microstructure"

    # Target data paths
    targets = {
        "basis": ms_path / "basis" / "gross_basis_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "irr": ms_path / "irr" / "irr_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
    }

    # Feature data paths
    features = {
        # Mid Prices
        # "fut_mid_price": ms_path / "mid-price" / "fut_mid_price_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_mid_price": ms_path / "mid-price" / "ctd_mid_price_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Micro Prices
        # "fut_micro_price": ms_path / "micro-price" / "fut_micro_price_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_micro_price": ms_path / "micro-price" / "ctd_micro_price_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Spreads
        # "fut_spread": ms_path / "spread" / "fut_spread_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_spread": ms_path / "spread" / "ctd_spread_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Order Book Imbalance
        # "fut_obi": ms_path / "obi" / "fut_obi_lvl_1_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_obi": ms_path / "obi" / "ctd_obi_lvl_1_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_obi_lvl_2": ms_path / "obi" / "fut_obi_lvl_2_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_obi_lvl_2": ms_path / "obi" / "ctd_obi_lvl_2_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_obi_lvl_3": ms_path / "obi" / "fut_obi_lvl_3_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_obi_lvl_3": ms_path / "obi" / "ctd_obi_lvl_3_scaled_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # Order Flow Imbalance
        # "fut_bof": ms_path / "ofi" / "fut_bof_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bof": ms_path / "ofi" / "ctd_bof_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_bof_lvl_2": ms_path / "ofi" / "fut_bof_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bof_lvl_2": ms_path / "ofi" / "ctd_bof_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_bof_lvl_3": ms_path / "ofi" / "fut_bof_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_bof_lvl_3": ms_path / "ofi" / "ctd_bof_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_aof": ms_path / "ofi" / "fut_aof_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_aof": ms_path / "ofi" / "ctd_aof_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "fut_aof_lvl_2": ms_path
        / "ofi"
        / "fut_aof_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "ctd_aof_lvl_2": ms_path
        / "ofi"
        / "ctd_aof_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "fut_aof_lvl_3": ms_path
        / "ofi"
        / "fut_aof_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        "ctd_aof_lvl_3": ms_path
        / "ofi"
        / "ctd_aof_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_ofi": ms_path / "ofi" / "fut_ofi_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_ofi": ms_path / "ofi" / "ctd_ofi_lvl_1_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_ofi_lvl_2": ms_path / "ofi" / "fut_ofi_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_ofi_lvl_2": ms_path / "ofi" / "ctd_ofi_lvl_2_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "fut_ofi_lvl_3": ms_path / "ofi" / "fut_ofi_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
        # "ctd_ofi_lvl_3": ms_path / "ofi" / "ctd_ofi_lvl_3_freq_1s_from_2022_08_01_to_2024_12_30.parquet",
    }

    return DataPathsConfig(ticker=ticker, targets=targets, features=features)
