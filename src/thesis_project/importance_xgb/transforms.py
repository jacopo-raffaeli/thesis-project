"""Feature transform classes for preprocessing pipeline."""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_first_valid(series: pd.Series) -> float:
    """
    Get first non-zero, non-NaN value in a series, to compute valid ratio features.

    # Args:
        - series

    # Return:
        - first valid value in the series
    """
    if len(series) == 0:
        raise ValueError("The series is empty")

    if series.isna().all():
        raise ValueError("The series is all NaNs")

    if (series == 0).all():
        raise ValueError("The series has no non-zero values")

    filled = series.replace(0, np.nan).bfill()
    return filled.iloc[0]


class DeltaTransform:
    """
    Compute first differences of features within daily sessions.
    """

    def __init__(self, columns: List[str], deltas: List[int]):
        """
        # Args:
            - columns: List of columns where to compute first differences
            - time_deltas: List of time windows (in seconds) to use for first differences
        """
        if any(d <= 0 for d in deltas):
            raise ValueError("Time deltas must be positive integers")

        self.columns = columns
        self.time_deltas = deltas

    def compute(self, df: pd.DataFrame, grouped_by_session) -> pd.DataFrame:
        """
        Compute delta features grouped by trading daily sessions.

        # Args:
            - df: pd.DataFrame containing the data to use
            - grouped_by_session: output of df.groupby(df.index.normalize())

        # Return:
            - df with same index as the input one and transformed features
        """
        features = {}

        for col in self.columns:
            if col not in df.columns:
                logger.warning("Column %s not found in DataFrame, skipping", col)
                continue

            gcol = grouped_by_session[col]

            for delta in self.time_deltas:
                col_name = f"{col}_delta_{delta}s"
                features[col_name] = gcol.diff(delta)
                # nan_perc = features[col_name].isna().mean() * 100
                # logger.debug("Computed: %s (%.2f%% NaN)", col_name, nan_perc)

        return pd.DataFrame(features, index=df.index)


class RollingTransform:
    """
    Compute rolling features within daily sessions.
    """

    def __init__(self, columns: List[str], windows: List[int], stats: List[str]):
        """
        # Args:
            - columns: List of columns where to compute first differences
            - windows: List of time windows to use for the rolling features
            - stats: List of stats to use for the rolling windwos (must be compatible with pandas rolling stats)
        """
        if any(w <= 0 for w in windows):
            raise ValueError("Windows must be positive integers")

        self.columns = columns
        self.windows = windows
        self.stats = stats

    def compute(self, df: pd.DataFrame, grouped_by_session) -> pd.DataFrame:
        """
        Compute rolling features grouped by trading daily sessions.

        # Args:
            - df: pd.DataFrame containing the data to use
            - grouped_by_session: output of df.groupby(df.index.normalize())

        # Return:
            - df with same index as the input one and transformed features
        """
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
                        # nan_perc = features[col_name].isna().mean() * 100
                        # logger.debug("   - Computed: %s (%.2f%% NaN)", col_name, nan_perc)
                    except AttributeError:
                        logger.warning(
                            "   - Failed: %s, %s not available in rolling object", col_name, stat
                        )
                        continue

        return pd.DataFrame(features, index=df.index)


class RatioTransform:
    """Compute ratio features as x_t / reference_value."""

    def __init__(self, columns: list[str], references: list[str]):
        """
        # Args:
            - columns: List of columns where to compute first differences
            - references: List of reference values to use, currently available:
                - "day_start"
                - "hour_start"
        """
        if isinstance(references, str):
            references = [references]

        self.columns = columns
        self.references = references

    def compute(self, df: pd.DataFrame, grouped_by_session) -> pd.DataFrame:
        """
        Compute ratio features grouped by trading daily sessions.

        # Args:
            - df: pd.DataFrame containing the data to use
            - grouped_by_session: output of df.groupby(df.index.normalize())

        # Return:
            - df with same index as the input one and transformed features
        """
        features = {}

        for col in self.columns:
            if col not in df.columns:
                logger.warning("   - Skipped: %s not found in DataFrame", col)
                continue

            for reference in self.references:
                col_name = f"{col}_{reference}_ratio"

                try:
                    if reference == "day":
                        first_values = grouped_by_session[col].transform(_get_first_valid)

                    elif reference == "hour":
                        results = []
                        for _, day_group in grouped_by_session:
                            hourly_first = (
                                day_group[col]
                                .groupby(day_group.index.hour)
                                .transform(_get_first_valid)
                            )
                            results.append(hourly_first)
                        first_values = pd.concat(results)

                    ratio = df[col] / first_values
                    features[col_name] = ratio
                    # nan_perc = features[col_name].isna().mean() * 100
                    # logger.debug("   - Computed: %s (%.2f%% NaN)", col_name, nan_perc)

                except Exception as e:
                    logger.warning(
                        "Failed: %s with reference %s: %s",
                        col_name,
                        reference,
                        str(e),
                    )
                    continue

        return pd.DataFrame(features, index=df.index)
