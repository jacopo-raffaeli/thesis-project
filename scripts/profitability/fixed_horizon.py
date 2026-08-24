import argparse
from typing import get_args

from thesis_project.config import FutTicker
from thesis_project.profitability import (
    fixed_horizon,
    settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        type=str,
        choices=get_args(FutTicker),
        required=True,
        help="Ticker for which to execute the fixed profit analysis",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # fmt: off
    ANALYSES = [
        # settings.AnalysisConfig(
        #     price_mode="mid",
        #     fut_contract_mode="frac",
        #     volume_modes=("ignore", "level"),
        # ),
        # settings.AnalysisConfig(
        #     price_mode="mid",
        #     fut_contract_mode="round",
        #     volume_modes=("ignore", "level"),
        # ),
        # settings.AnalysisConfig(
        #     price_mode="quoted",
        #     fut_contract_mode="frac",
        #     volume_modes=("ignore", "level"),
        # ),
        settings.AnalysisConfig(
            price_mode="quoted",
            fut_contract_mode="round",
            volume_modes=("ignore", "level"),
        ),
    ]

    HORIZONS = [
        10,
        30,
        1*60,
        5*60,
        10*60,
        # 15*60,
        30*60,
        # 45*60,
        1*60**2,
        2*60**2,
        # 3*60**2,
        # 24*60**2,
        # 3*24*60**2,
        # 5*24*60**2
    ]

    CONFIG = fixed_horizon.FixedHorizonConfig(
        ticker=args.ticker,
        ctd_contracts=1,
        horizons=HORIZONS,
        tolerance=0,
        analyses=ANALYSES,
    )
    # fmt: on

    fixed_horizon.run_fixed_horizon(CONFIG)


if __name__ == "__main__":
    main()
