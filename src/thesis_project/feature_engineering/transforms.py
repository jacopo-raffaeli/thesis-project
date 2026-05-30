"""Feature transform classes for preprocessing pipeline."""

import logging
from math import nan
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def get_first_valid(series: pd.Series) -> float:
    """Get first non-zero, non-NaN value, backward-filling to find it."""
    if len(series) == 0:
        return nan
    filled = series.replace(0, nan).bfill()
    return filled.iloc[0]


class DeltaTransform:
    """Compute differenced features within daily sessions."""

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
    """Compute rolling features within daily sessions."""

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


class RatioTransform:
    """Compute ratio features x_t / reference_value."""

    def __init__(self, columns: list[str], references: list[str]):
        """Initialize with columns and references."""
        self.columns = columns
        self.references = references
        valid_references = {"day_start", "hour_start"}
        for reference in self.references:
            if reference not in valid_references:
                raise ValueError(f"Invalid reference: {reference}. Valid: {valid_references}")

    def compute(self, df: pd.DataFrame, grouped_by_session) -> pd.DataFrame:
        """Compute ratio features within daily session."""
        features = {}

        for col in self.columns:
            if col not in df.columns:
                logger.warning("   - Skipped: %s not found in DataFrame", col)
                continue

            for reference in self.references:
                col_name = f"{col}_ratio_{reference}"

                try:
                    if reference == "day_start":
                        first_values = grouped_by_session[col].transform(get_first_valid)

                    elif reference == "hour_start":
                        results = []
                        for _, day_group in grouped_by_session:
                            hourly_first = (
                                day_group[col]
                                .groupby(day_group.index.hour)
                                .transform(get_first_valid)
                            )
                            results.append(hourly_first)
                        first_values = pd.concat(results)

                    ratio = df[col] / first_values
                    features[col_name] = ratio
                    nan_perc = features[col_name].isna().mean() * 100
                    logger.debug("   - Computed: %s (%.2f%% NaN)", col_name, nan_perc)
                except Exception as e:
                    logger.warning(
                        "   - Failed: %s with reference %s: %s",
                        col_name,
                        reference,
                        str(e),
                    )
                    continue

        return pd.DataFrame(features, index=df.index)
