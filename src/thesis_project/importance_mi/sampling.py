"""Sampling strategies for data selection."""

import logging
from typing import Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def sample_data(
    data: pd.DataFrame,
    strategy: str,
    sampling_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """Sample data using specified strategy (uniform_v1)."""
    if sampling_params is None:
        sampling_params = {}

    strategies: Dict[str, Callable] = {
        "uniform_v1": lambda: _uniform_v1(data, sampling_params),
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. " f"Available: {list(strategies.keys())}")

    return strategies[strategy]()


def _uniform_v1(data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Random uniform sampling by sample_size or sample_ratio."""
    random_state = params.get("random_state", None)

    # Determine sample size
    has_size = "sample_size" in params
    has_ratio = "sample_ratio" in params

    if has_size and has_ratio:
        raise ValueError("Cannot specify both sample_size and sample_ratio")
    if not has_size and not has_ratio:
        raise ValueError("Must specify either sample_size or sample_ratio")

    if has_size:
        sample_size = params["sample_size"]
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError(f"sample_size must be positive int, got {sample_size}")
        if sample_size > len(data):
            raise ValueError(f"sample_size ({sample_size}) exceeds data length ({len(data)})")
        n = sample_size
    else:
        sample_ratio = params["sample_ratio"]
        if not (0 < sample_ratio <= 1):
            raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
        n = max(1, int(len(data) * sample_ratio))

    return data.sample(n=n, random_state=random_state, replace=False)
