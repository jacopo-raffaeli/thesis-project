import argparse
from typing import get_args

from thesis_project.config import FutTicker
from thesis_project.profitability import (
    fixed_profit,
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
        settings.AnalysisConfig(
            price_mode="mid",
            fut_contract_mode="frac",
            volume_modes=("ignore", "level"),
        ),
        settings.AnalysisConfig(
            price_mode="mid",
            fut_contract_mode="round",
            volume_modes=("ignore", "level"),
        ),
        settings.AnalysisConfig(
            price_mode="quoted",
            fut_contract_mode="frac",
            volume_modes=("ignore", "level"),
        ),
        settings.AnalysisConfig(
            price_mode="quoted",
            fut_contract_mode="round",
            volume_modes=("ignore", "level"),
        ),
    ]

    PROFITS = [
        100.,
        500.,
        1000.
    ]

    CONFIG = fixed_profit.FixedProfitConfig(
        ticker=args.ticker,
        ctd_contracts=1,
        profits=PROFITS,
        max_holding_time=None,
        analyses=ANALYSES,
        n_jobs=len(ANALYSES),
    )
    # fmt: on

    fixed_profit.run_fixed_profit(CONFIG)


if __name__ == "__main__":
    main()
