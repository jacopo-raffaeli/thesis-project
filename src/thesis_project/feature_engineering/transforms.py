"""Feature transform classes for preprocessing pipeline."""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class DeltaTransform:
    """Compute time-delta features within daily sessions."""

    def __init__(self, columns: List[str], time_deltas: List[int]):
        """Initialize with columns and time delta values (seconds)."""
        self.columns = columns
        self.time_deltas = time_deltas

    def compute(self, df: pd.DataFrame, grouped_by_session) -> pd.DataFrame:
        """Compute delta features grouped by trading day."""
        features = {}

        for col in self.columns:
            if col not in df.columns:
                logger.warning("   - Column %s not found in DataFrame, skipping", col)
                continue

            gcol = grouped_by_session[col]

            for delta in self.time_deltas:
                col_name = f"{col}_delta_{delta}s"
                features[col_name] = gcol.diff(delta)
                nan_perc = features[col_name].isna().mean() * 100
                logger.debug("   - Computed: %s (%.2f%% NaN)", col_name, nan_perc)

        return pd.DataFrame(features, index=df.index)


class RollingTransform:
    """Compute rolling statistics within daily sessions."""

    def __init__(self, columns: List[str], windows: List[int], stats: List[str]):
        """Initialize with columns, windows (seconds), and stats."""
        self.columns = columns
        self.windows = windows
        self.stats = stats

    def compute(self, df: pd.DataFrame, grouped_by_session) -> pd.DataFrame:
        """Compute rolling statistics grouped by trading day."""
        features = {}

        for col in self.columns:
            if col not in df.columns:
                logger.warning("   - Skipped: %s not found in DataFrame", col)
                continue

            gcol = grouped_by_session[col]

            for window in self.windows:
                min_periods = window // 2
                rolled = gcol.rolling(window=window, min_periods=min_periods)

                for stat in self.stats:
                    try:
                        col_name = f"{col}_roll_{stat}_{window}s"
                        result = getattr(rolled, stat)()
                        features[col_name] = result.droplevel(0)
                        nan_perc = features[col_name].isna().mean() * 100
                        logger.debug("   - Computed: %s (%.2f%% NaN)", col_name, nan_perc)
                    except AttributeError:
                        logger.warning(
                            "   - Failed: %s, %s not available in rolling object", col_name, stat
                        )
                        continue

        return pd.DataFrame(features, index=df.index)
