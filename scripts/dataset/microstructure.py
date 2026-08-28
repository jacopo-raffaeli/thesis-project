import argparse
from functools import partial
from typing import Callable, get_args

import pandas as pd

from thesis_project import config
from thesis_project.dataset import basis, irr, microstructure
from thesis_project.utils.logging import setup_logging

# fmt: off
MICROSTRUCTURE_FEATURES: dict[tuple[str, str], Callable[[pd.DataFrame], pd.Series]] = {
    # Prices
    **{
        ("price", f"{side}_price_{level}"): partial(microstructure.get_price, level=level, side=side)
        for level in config.LOB.levels
        for side in get_args(config.LobSide)
    },
    # Sizes
    **{
        ("size", f"{side}_size_{level}"): partial(microstructure.get_size, level=level, side=side)
        for level in config.LOB.levels
        for side in get_args(config.LobSide)
    },
    # Mid Price
    **{
        ("mid_price", "mid_price"): microstructure.compute_mid_price
    },
    # Micro Price
    **{
        ("micro_price", "micro_price"): microstructure.compute_micro_price
    },
    # OBI
    **{
        ("obi", f"obi_{max_level}"): partial(microstructure.compute_obi, max_level=max_level, ratio=True)
        for max_level in [1, 2, 3]
    },
    # bOF
    **{
        ("ofi", f"bof_{level}"): partial(microstructure.compute_bof, level=level)
        for level in [1, 2, 3]
    },
    # aOF
    **{
        ("ofi", f"aof_{level}"): partial(microstructure.compute_aof, level=level)
        for level in [1, 2, 3]
    },
    # OFI
    **{
        ("ofi", f"ofi_{level}"): partial(microstructure.compute_ofi, level=level)
        for level in [1, 2, 3]
    },
    # Slope
    **{
        ("slope", f"{side}_slope_{microstructure.SLOPE_DICT[slope_type]}_{max_level}"): partial(microstructure.compute_slope, max_level=max_level, side=side, slope_type=slope_type)
        for max_level in [1, 2, 3]
        for side in get_args(config.LobSide)
        for slope_type in get_args(microstructure.SlopeType)
    }
}
# fmt: on


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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

    microstructure.microstructure_per_ticker(
        ticker=args.ticker,
        par_funcs=MICROSTRUCTURE_FEATURES,
    )

    root = config.DATA_PRO_DIR / str(args.ticker) / "microstructure"

    # Compute gross basis
    s = basis.compute_gross_basis(ticker=args.ticker, mode="mid")
    filename = "basis"
    path = root / filename / filename
    path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame(filename).rename_axis(config.LOB.index_name).to_parquet(path)

    # Compute implied repo rate
    s = irr.compute_irr(ticker=args.ticker, mode="mid")
    filename = "irr"
    path = root / filename / filename
    path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame(filename).rename_axis(config.LOB.index_name).to_parquet(path)


if __name__ == "__main__":
    main()
