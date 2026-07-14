"""Metric computation for information-theoretic analysis."""

import logging
from typing import Dict, List, Optional

import infomeasure as im
import numpy as np

logger = logging.getLogger(__name__)


def compute_metrics(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray],
    metric_names: List[str],
    metrics_params: Dict,
) -> Dict[str, float]:
    """
    Compute requested metrics for (x, y, z) triplet.
    Raises ValueError if conditional metrics requested without z.
    """

    # Check if conditioning signal is needed for requested metrics
    requested_conditional = {"cmi", "cte"} & set(metric_names)
    if requested_conditional and (z is None or len(z) == 0):
        raise ValueError(f"Conditioning signal required for metrics: {requested_conditional}")

    # Define metrics
    metrics = {
        "pc": lambda p: compute_pc(x, y),
        "mi": lambda p: compute_mi(x, y, **p),
        "cmi": lambda p: compute_cmi(x, y, z, **p),  # type: ignore
        "te": lambda p: compute_te(x, y, **p),
        "cte": lambda p: compute_cte(x, y, z, **p),  # type: ignore
    }

    # Compute only requested metrics
    results = {}
    for name in metric_names:
        if name not in metrics:
            raise KeyError(f"Unknown metric: {name}. Available: {list(metrics.keys())}")
        params = metrics_params.get(name, {})
        results[name] = metrics[name](params)

    return results


def compute_pc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson Correlation."""
    return float(np.corrcoef(x, y)[0, 1])


def compute_mi(x: np.ndarray, y: np.ndarray, **params) -> float:
    """Compute Mutual Information."""
    return float(im.mutual_information(x, y, **params))


def compute_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray, **params) -> float:
    """Compute Conditional Mutual Information."""
    return float(im.conditional_mutual_information(x, y, cond=z, **params))


def compute_te(x: np.ndarray, y: np.ndarray, **params) -> float:
    """Compute Transfer Entropy."""
    return float(im.transfer_entropy(x, y, **params))


def compute_cte(x: np.ndarray, y: np.ndarray, z: np.ndarray, **params) -> float:
    """Compute Conditional Transfer Entropy."""
    return float(im.conditional_transfer_entropy(x, y, cond=z, **params))
