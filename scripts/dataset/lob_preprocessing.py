import argparse
from typing import get_args

from thesis_project.config import FutTicker
from thesis_project.dataset import lob


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        type=str,
        choices=get_args(FutTicker),
        required=True,
        help="Ticker for which to execute the preprocessing pipeline",
    )

    args = parser.parse_args()

    lob.preprocess_all_lobs(ticker=args.ticker)


if __name__ == "__main__":
    main()
