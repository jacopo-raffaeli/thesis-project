import argparse
from typing import get_args

from thesis_project import config, utils
from thesis_project.dataset.criticalities import DEFAULT_CHECKS, Settings, find_criticalities

# TODO: Define default settings for fbts

DEFAULT_SETTINGS: dict[config.FutTicker, Settings] = {
    "fbtp": Settings(
        checks=DEFAULT_CHECKS,
        min_time=config.STD_OPENING_TIME,
        max_time=config.STD_CLOSING_TIME,
        nan_threshold=0.25 * 100,
        sec_threshold=60.0,
        relevant_columns=tuple(config.LOB.get_columns(levels=[1, 2, 3])),
    ),
    "fbts": Settings(
        checks=DEFAULT_CHECKS,
        min_time=config.STD_OPENING_TIME,
        max_time=config.STD_CLOSING_TIME,
        nan_threshold=0.25 * 100,
        sec_threshold=60.0,
        relevant_columns=tuple(config.LOB.get_columns(levels=[1, 2, 3])),
    ),
}


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


def main() -> None:
    args = parse_args()

    utils.logging.setup_logging(level=args.log_level)

    result = find_criticalities(args.ticker, DEFAULT_SETTINGS[args.ticker])
    path = config.DATA_PRO_DIR / args.ticker / "criticalities.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)


if __name__ == "__main__":
    main()
