import argparse
from typing import get_args

from thesis_project import config, eda, utils
from thesis_project.utils.misc import get_dates_to_exclude


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        type=str,
        choices=get_args(config.FutTicker),
        required=True,
        help="Ticker for which to execute the stationarity tests",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Series for which to execute the stationarity tests",
    )

    parser.add_argument(
        "--window",
        type=str,
        choices=["daily", "hourly"],
        required=True,
        help="Time frequency on which to perform the stationarity tests",
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


FREQ_LAGS_DICT_DAILY: dict[str, list[int]] = {
    "1s": [60]
    # "1s": [10, 30, 60, 300, 600, 900, 1800, 3600, 7200],
}

FREQ_LAGS_DICT_HOURLY: dict[str, list[int]] = {
    "1s": [10, 30, 60, 300, 600, 900, 1800],
}

FREQ_LAGS_DICT: dict[str, dict[str, list[int]]] = {
    "daily": FREQ_LAGS_DICT_DAILY,
    "hourly": FREQ_LAGS_DICT_HOURLY,
}


def main():
    args = parse_args()

    utils.logging.setup_logging(level=args.log_level)

    dates_to_exclude = get_dates_to_exclude(args.ticker, config.DEFAULT_EXCLUDED_DATES)

    eda.stationarity.main(
        ticker=args.ticker,
        name=args.name,
        window=args.window,
        freq_lags_dict=FREQ_LAGS_DICT[args.window],
        dates_to_exclude=dates_to_exclude,
    )


if __name__ == "__main__":
    main()
