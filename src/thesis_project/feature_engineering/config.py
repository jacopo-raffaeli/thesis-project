from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from thesis_project import config as global_config


@dataclass
class AnalysisConfig:
    """
    Configuration for feature engineering analysis.
    """

    ticker: str = ""

    min_time: str = "09:00:00"
    max_time: str = "17:00:00"

    sampling_strategy: str = "uniform_v1"
    sample_ratio: float = 0.25

    # Available metrics
    metrics: List[str] = field(
        default_factory=lambda: [
            "pc",  # pearson correlation
            "mi",  # mutual information
            "cmi",  # conditional mutual information
            "te",  # transfer entropy
            "cte",  # conditional transfer entropy
        ]
    )

    # Based on infomeasure's API settings
    metrics_params = {
        "pc": {},
        "mi": {
            "approach": "ksg",
            "k": 5,
        },
        "cmi": {
            "approach": "ksg",
            "k": 4,
        },
        "te": {
            "approach": "ksg",
            "k": 5,
        },
        "cte": {
            "approach": "ksg",
            "k": 4,
        },
    }

    seed: int = 42
    n_jobs: int = -1
    checkpoint_interval: int = 100

    # output_path is set by create_config() based on ticker
    output_path: Path | None = None

    date_filters_offsets: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "fut_last_trading_days": [5, 0],
        }
    )

    analysis_horizons: List[int] = field(
        default_factory=lambda: [
            60,
            600,
        ]
    )

    # Drop base series after processing the derived ones
    drop_base_targets: bool = False
    drop_base_features: bool = False

    # Transform configurations
    target_transforms: Dict[str, Dict] = field(
        default_factory=lambda: {
            "delta": {
                "time_deltas": [
                    10,
                    60,
                    300,
                ],
            },
        }
    )

    feature_transforms: Dict[str, Dict] = field(
        default_factory=lambda: {
            "delta": {
                "time_deltas": [
                    10,
                    60,
                    300,
                ],
            },
            "rolling": {
                "windows": [
                    300,
                    600,
                    1800,
                ],
                "stats": [
                    "mean",
                    "std",
                    "min",
                    "max",
                ],
            },
        }
    )


def create_config(**overrides) -> AnalysisConfig:
    """
    Create config with CLI overrides.

    Args:
        **overrides: Configuration overrides. Must include 'ticker'.

    Returns:
        AnalysisConfig with output_path derived from ticker.

    Raises:
        ValueError: If ticker not provided in overrides.
    """
    # Validate ticker is provided
    if "ticker" not in overrides:
        raise ValueError("ticker must be provided in config overrides")

    ticker = overrides["ticker"]

    # Create base config
    config = AnalysisConfig()

    # Apply all overrides
    for key, val in overrides.items():
        if hasattr(config, key):
            setattr(config, key, val)

    # Set output_path based on ticker
    config.output_path = (
        global_config.INT_DIR / ticker / "feature-engineering" / "importance-analysis"
    )
    assert config.output_path is not None
    if not config.output_path.exists():
        config.output_path.mkdir(parents=True, exist_ok=True)

    return config
