import argparse
from typing import get_args

from thesis_project import config, dataset
from thesis_project.utils.logging import setup_logging


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

    basis = dataset.basis.compute_gross_basis(ticker=args.ticker, mode="mid")
    filename = "gross_basis_mid"

    path = config.DATA_PRO_DIR / str(args.ticker) / "microstructure" / "basis" / filename
    path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    basis.to_frame(filename).rename_axis(config.LOB.index_name).to_parquet(path)


if __name__ == "__main__":
    main()
