"""Feature importance analysis with xgboost runner"""

import argparse
import sys

from thesis_project.importance_xgb import config, orchestration, utils


def parse_args():
    parser = argparse.ArgumentParser(description="Future importance analysis with xgboost")
    parser.add_argument(
        "--ticker", type=str, choices=["fbtp", "fbts"], help="Ticker symbol", required=True
    )
    parser.add_argument("--sample-ratio", type=float, help="Sampling ratio for data", required=True)
    parser.add_argument(
        "--n_quantile", type=int, help="Number of quantiles for the classification task", default=3
    )
    parser.add_argument(
        "--n-jobs-preprocessing",
        type=int,
        help="Parallel jobs for feature preprocessing",
        required=True,
    )
    parser.add_argument("--n-jobs-xgb", type=int, help="Parallel jobs for XGBoost", required=True)
    parser.add_argument(
        "--optuna-n-trials",
        type=int,
        help="Number of optimization trials for Optuna",
        required=True,
    )
    parser.add_argument("--use-base-target", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-tscv", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--analysis",
        type=str,
        help="Choose between importance and selection analysis",
        choices=["importance", "selection"],
        required=True,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
        default="INFO",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger = utils.setup_logging(level=args.log_level)

    try:
        overrides = {k: v for k, v in vars(args).items() if v is not None and k != "log_level"}
        config_obj = config.create_config(**overrides)
        orchestration.run_analysis(config_obj)

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
