"""Feature engineering analysis runner."""

import argparse
import sys

from thesis_project.feature_engineering import (
    run_analysis,
    setup_logging,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Engineering Analysis")
    parser.add_argument(
        "--ticker",
        default="fbtp",
        choices=["fbtp", "fbts"],
        help="Ticker symbol (default: fbtp)",
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Metrics to compute: pc, mi, cmi, te, cte (comma-separated, required)",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Sampling ratio for data (default: 0.1 from config, None to disable sampling)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (-1 for all cores, default: -1 from config)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Checkpoint every N results (default: 100 from config, -1 to disable)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show progress bar and detailed output (default: True)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup console-only logging (file logging added in orchestration after run_dir created)
    logger = setup_logging(level=args.log_level)

    # Parse metrics argument
    try:
        metrics = [m.strip() for m in args.metrics.split(",")]
    except ValueError as e:
        print(f"Error parsing metrics: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate metrics
    valid_metrics = ["pc", "mi", "cmi", "te", "cte"]
    for metric in metrics:
        if metric not in valid_metrics:
            print(f"Error: Unknown metric '{metric}'. Valid: {valid_metrics}", file=sys.stderr)
            sys.exit(1)

    # Build config overrides from CLI arguments (only what user provided)
    from thesis_project.feature_engineering.config import create_config

    config_overrides = {
        "ticker": args.ticker,
        "metrics": metrics,
    }

    # Add optional overrides only if user specified them
    if args.sample_ratio is not None:
        config_overrides["sample_ratio"] = args.sample_ratio
    if args.n_jobs is not None:
        config_overrides["n_jobs"] = args.n_jobs
    if args.checkpoint_interval is not None:
        config_overrides["checkpoint_interval"] = args.checkpoint_interval

    config = create_config(**config_overrides)

    try:
        run_analysis(config)

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)

        return 1


if __name__ == "__main__":
    sys.exit(main())
