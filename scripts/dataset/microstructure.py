import argparse
from functools import partial
from typing import Callable, get_args

import pandas as pd

from thesis_project import config
from thesis_project.dataset import cross_asset_basis, cross_asset_irr, lob_microstructure
from thesis_project.utils.logging import setup_logging

# fmt: off
MICROSTRUCTURE_FEATURES: dict[tuple[str, str], Callable[[pd.DataFrame], pd.Series]] = {
    # Prices
    **{
        ("price", f"{side}_price_{level}"): partial(lob_microstructure.get_price, level=level, side=side)
        for level in config.LOB.levels
        for side in get_args(config.LobSide)
    },
    # Sizes
    **{
        ("size", f"{side}_size_{level}"): partial(lob_microstructure.get_size, level=level, side=side)
        for level in config.LOB.levels
        for side in get_args(config.LobSide)
    },
    # Mid Price
    **{
        ("mid-price", "mid_price"): lob_microstructure.compute_mid_price
    },
    # Spread
    **{
        ("spread", "spread"): lob_microstructure.compute_spread
    },
    # Micro Price
    **{
        ("micro-price", "micro_price"): lob_microstructure.compute_micro_price
    },
    # OBI
    **{
        ("obi", f"obi_{max_level}"): partial(lob_microstructure.compute_obi, max_level=max_level, ratio=ratio)
        for max_level in [1, 2, 3]
        for ratio in [True]
    },
    # bOF
    **{
        ("ofi", f"bof_{level}"): partial(lob_microstructure.compute_bof, level=level)
        for level in [1, 2, 3]
    },
    # aOF
    **{
        ("ofi", f"aof_{level}"): partial(lob_microstructure.compute_aof, level=level)
        for level in [1, 2, 3]
    },
    # OFI
    **{
        ("ofi", f"ofi_{level}"): partial(lob_microstructure.compute_ofi, level=level)
        for level in [1, 2, 3]
    },
    # Slope
    **{
        ("slope", f"{side}_slope_{max_level}_{lob_microstructure.SLOPE_DICT[slope_type]}"): partial(lob_microstructure.compute_slope, max_level=max_level, side=side, slope_type=slope_type)
        for max_level in [1, 2, 3]
        for side in get_args(config.LobSide)
        for slope_type in get_args(lob_microstructure.SlopeType)
    }
}

CROSS_ASSET: dict[tuple[str, str], Callable[[config.FutTicker], pd.Series]] = {
    ("basis", "basis"): partial(cross_asset_basis.compute_basis, mode="mid"),
    ("irr", "irr"): partial(cross_asset_irr.compute_irr, mode="mid"),
}
# fmt: on


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script to easily generate all the time series used on the project for a specified ticker"
    )

    parser.add_argument(
        "--ticker",
        type=str,
        choices=get_args(config.FutTicker),
        required=True,
        help="Ticker for which to execute the preprocessing pipeline",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ),
        default="INFO",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    setup_logging(level=args.log_level)

    if len(MICROSTRUCTURE_FEATURES) > 0:
        lob_microstructure.microstructure_per_ticker(
            ticker=args.ticker,
            par_funcs=MICROSTRUCTURE_FEATURES,
        )

    if len(CROSS_ASSET) > 0:
        lob_microstructure.cross_asset_per_ticker(ticker=args.ticker, par_funcs=CROSS_ASSET)


if __name__ == "__main__":
    main()
