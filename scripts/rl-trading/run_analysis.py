import argparse

from thesis_project import rl, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--ticker",
    #     type=str,
    #     choices=get_args(config.FutTicker),
    #     required=True,
    #     help="Ticker for which to execute the preprocessing pipeline",
    # )

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

    utils.logging.setup_logging(level=args.log_level)

    rl.train_trading.main()


if __name__ == "__main__":
    main()
