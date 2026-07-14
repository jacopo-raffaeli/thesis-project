"""Sampling strategies for data selection."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from thesis_project.importance_xgb import registry

logger = logging.getLogger(__name__)


def sample_timestamps(
    idx: pd.DatetimeIndex,
    strategy: str,
    sampling_params: Optional[Dict] = None,
) -> pd.DatetimeIndex:
    """
    Sample data using specified strategy.

    # Args:
        - idx: Index to sample from
        - strategy: Sample strategy, currently available:
            - uniform_v1
        - sampling_params: Strategy specific parameters

    # Returns:
        - Index sampled according to the strategy selected and sorted
    """
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Index must be a pd.DatetimeIndex")

    if sampling_params is None:
        sampling_params = {}

    if strategy not in registry.SAMPLING_STRATEGIES_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available: {sorted(registry.SAMPLING_STRATEGIES_REGISTRY)}"
        )

    sampling_function = registry.SAMPLING_STRATEGIES_REGISTRY[strategy]
    idx_sampled = sampling_function(idx, sampling_params)

    if not idx_sampled.is_monotonic_increasing:
        idx_sampled = idx_sampled.sort_values()
    logger.info("Index sampled with strategy %s", strategy)

    return idx_sampled


def _uniform_v1(idx: pd.DatetimeIndex, params: Dict) -> pd.DatetimeIndex:
    """
    Random uniform sampling by ratio.

    # Args:
        - idx: Index to sample from

    # Returns:
        - Index sampled according to the strategy selected
    """
    random_state = params.get("random_state", None)

    has_ratio = "sample_ratio" in params
    if not has_ratio:
        raise ValueError("Must specify sample_ratio")

    sample_ratio = params["sample_ratio"]
    if not (0 < sample_ratio <= 1):
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")

    n = max(1, int(len(idx) * sample_ratio))
    rng = np.random.default_rng(random_state)
    positions = rng.choice(len(idx), size=n, replace=False)

    return idx.take(positions)
