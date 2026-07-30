from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy


# Base transform class
@dataclass(frozen=True)
class BaseTransform(ABC):
    lags: list[int] | int | None = None
    keep_original: bool = True

    def __post_init__(self):
        lags = _init_data(self.lags, _normalize_lags, _validate_lags)
        object.__setattr__(self, "lags", lags)

        assert isinstance(self.lags, list)
        if not self.keep_original and len(self.lags) == 0:
            raise ValueError("Invalid transform: no base feature and no lags present")

    @abstractmethod
    def compute(self, series: pd.Series, grouped: SeriesGroupBy) -> dict[str, pd.Series]:
        """
        Compute transformed features
        """

    @abstractmethod
    def _transform_name(self, base_id: str, *args) -> str:
        """
        Return transform name
        """

    def _lag_name(self, base_id: str, lag: int) -> str:
        """
        Return lagged name
        """
        return f"{base_id}_lag_{lag}s"

    def _apply_lags(
        self, series: pd.Series, grouped: SeriesGroupBy, id: str
    ) -> dict[str, pd.Series]:
        """
        Lag trasnformed outputs
        """
        assert isinstance(self.lags, list)

        if not self.has_lags:
            raise ValueError("The transform object has no lags")

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("The series index is not a DatetimeIndex")

        lagged = {}
        for lag in self.lags:
            name = self._lag_name(id, lag)
            lagged[name] = grouped.shift(lag)

        return lagged

    def transform(self, base: pd.Series, grouped: SeriesGroupBy):
        """
        Perform transformation and lagging
        """
        out = {}

        transformed = self.compute(base, grouped)
        if self.keep_original:
            out.update(transformed)

        if self.has_lags:
            for name, transform in transformed.items():
                out.update(self._apply_lags(transform, grouped, name))

        return out

    @abstractmethod
    def base_output_names(self, base_id: str) -> list[str]:
        """
        Output list of non-lagged transform names
        """

    def output_names(self, base_id: str) -> list[str]:
        """
        Output list of transform names
        """
        assert isinstance(self.lags, list)
        base_names = self.base_output_names(base_id)
        names = []
        for base_name in base_names:
            if self.keep_original:
                names.append(base_name)

            for lag in self.lags:
                names.append(self._lag_name(base_name, lag))

        return names

    @property
    @abstractmethod
    def n_base_outputs(self) -> int:
        """
        Number of non-lagged transformed features computed
        """

    @property
    def n_outputs(self) -> int:
        """
        Number of transformed features computed
        """
        assert isinstance(self.lags, list)
        multiplier = len(self.lags)

        if self.keep_original:
            multiplier += 1

        return self.n_base_outputs * multiplier

    @property
    def has_lags(self) -> bool:
        assert isinstance(self.lags, list)
        return len(self.lags) > 0


@dataclass(frozen=True)
class Identity(BaseTransform):
    def __post_init__(self):
        super().__post_init__()

    def compute(self, series: pd.Series, grouped: SeriesGroupBy) -> dict[str, pd.Series]:
        out = {}
        name = self._transform_name(str(series.name))
        out[name] = series
        return out

    @property
    def n_base_outputs(self) -> int:
        return 1

    def _transform_name(self, base_id: str) -> str:
        return base_id

    def base_output_names(self, base_id: str) -> list[str]:
        return [self._transform_name(base_id)]


@dataclass(frozen=True)
class Delta(BaseTransform):
    deltas: list[int] | int = field(default_factory=list[int])

    def __post_init__(self):
        super().__post_init__()

        deltas = _init_data(self.deltas, _normalize_deltas, _validate_deltas)
        object.__setattr__(self, "deltas", deltas)

    def compute(self, series: pd.Series, grouped: SeriesGroupBy) -> dict[str, pd.Series]:
        assert isinstance(self.deltas, list)

        out = {}
        for delta in self.deltas:
            name = self._transform_name(str(series.name), delta)
            out[name] = grouped.diff(delta)

        return out

    @property
    def n_base_outputs(self) -> int:
        assert isinstance(self.deltas, list)
        return len(self.deltas)

    def _transform_name(self, base_id: str, delta: int) -> str:
        return f"{base_id}_delta_{delta}s"

    def base_output_names(self, base_id: str) -> list[str]:
        assert isinstance(self.deltas, list)

        return [self._transform_name(base_id, delta) for delta in self.deltas]


@dataclass(frozen=True)
class Rolling(BaseTransform):
    VALID_STATS = ["mean", "std", "min", "max"]

    stats: list[str] | str = field(default_factory=list[str])
    windows: list[int] | int = field(default_factory=list[int])

    def __post_init__(self):
        super().__post_init__()

        stats = _init_data(self.stats, _normalize_stats, _validate_stats)
        object.__setattr__(self, "stats", stats)

        windows = _init_data(self.windows, _normalize_windows, _validate_windows)
        object.__setattr__(self, "windows", windows)

    def compute(self, series: pd.Series, grouped: SeriesGroupBy) -> dict[str, pd.Series]:
        assert isinstance(self.stats, list)
        assert isinstance(self.windows, list)

        out = {}
        for window in self.windows:
            rolled = grouped.rolling(window=window, min_periods=window // 2)
            for stat in self.stats:
                name = self._transform_name(str(series.name), stat, window)
                out[name] = getattr(rolled, stat)().droplevel(0)

        return out

    def base_output_names(self, base_id: str) -> list[str]:
        assert isinstance(self.stats, list)
        assert isinstance(self.windows, list)

        return [
            self._transform_name(base_id, stat, window)
            for stat in self.stats
            for window in self.windows
        ]

    @property
    def n_base_outputs(self) -> int:
        assert isinstance(self.stats, list)
        assert isinstance(self.windows, list)

        return len(self.stats) * len(self.windows)

    def _transform_name(self, base_id: str, stat: str, window: int) -> str:
        return f"{base_id}_roll_{stat}_{window}s"


@dataclass(frozen=True)
class Ratio(BaseTransform):
    VALID_REFERENCES = ["day", "hour"]

    references: list[str] | str = field(default_factory=list[str])

    def __post_init__(self):
        super().__post_init__()

        references = _init_data(self.references, _normalize_references, _validate_references)
        object.__setattr__(self, "references", references)

    def _get_first_valid(self, series: pd.Series) -> float:
        if len(series) == 0:
            raise ValueError("The series is empty")

        if series.isna().all():
            raise ValueError("The series is all NaNs")

        if (series == 0).all():
            raise ValueError("The series is all zeros")

        filled = series.replace(0, np.nan).bfill()
        return filled.iloc[0]

    def _day_reference(self, series: pd.Series, grouped: SeriesGroupBy) -> pd.Series:
        reference_values = grouped.transform(self._get_first_valid).reindex(series.index)

        return reference_values

    def _hour_reference(self, series: pd.Series, grouped: SeriesGroupBy) -> pd.Series:
        reference_values = []
        for _, day in grouped:
            temp = day.groupby(pd.Grouper(freq="h")).transform(self._get_first_valid)
            reference_values.append(temp)

        reference_values = pd.concat(reference_values).reindex(series.index)

        return reference_values

    def compute(self, series: pd.Series, grouped: SeriesGroupBy) -> dict[str, pd.Series]:
        assert isinstance(self.references, list)

        out = {}
        for reference in self.references:
            name = self._transform_name(str(series.name), reference)
            match reference:
                case "day":
                    reference_values = self._day_reference(series, grouped)

                case "hour":
                    reference_values = self._hour_reference(series, grouped)

                case _:
                    raise ValueError(f"Uknown ratio reference: {reference}")

            out[name] = series / reference_values

        return out

    def base_output_names(self, base_id: str) -> list[str]:
        assert isinstance(self.references, list)

        return [self._transform_name(base_id, reference) for reference in self.references]

    @property
    def n_base_outputs(self) -> int:
        return len(self.references)

    def _transform_name(self, base_id: str, reference: str) -> str:
        return f"{base_id}_{reference}_ratio"


def _init_data(value: Any, normalize: Callable, validate: Callable):
    value = _to_list(value)
    value = normalize(value)
    validate(value)

    return value


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def _normalize_lags(lags: list[int]) -> list[int]:
    return sorted(lags)


def _normalize_deltas(deltas: list[int]) -> list[int]:
    return sorted(deltas)


def _normalize_stats(stats: list[str]) -> list[str]:
    return stats


def _normalize_windows(windows: list[int]) -> list[int]:
    return sorted(windows)


def _normalize_references(references: list[str]) -> list[str]:
    return references


def _validate_lags(lags: list[int]) -> None:
    if any(type(lag) is not int for lag in lags):
        raise ValueError("Lags must be integers")

    if any(lag <= 0 for lag in lags):
        raise ValueError("Lags must be positive")

    if len(set(lags)) != len(lags):
        raise ValueError("Lags must not be duplicated")


def _validate_deltas(deltas: list[int]) -> None:
    if len(deltas) == 0:
        raise ValueError("Deltas must be non-empty")

    if any(type(delta) is not int for delta in deltas):
        raise ValueError("Deltas must be integers")

    if any(delta <= 0 for delta in deltas):
        raise ValueError("Deltas must be positive")

    if len(set(deltas)) != len(deltas):
        raise ValueError("Deltas must not be duplicated")


def _validate_stats(stats: list[str]):
    if len(stats) == 0:
        raise ValueError("Rolling stats must be non-empty")

    if any(stat not in Rolling.VALID_STATS for stat in stats):
        raise ValueError(f"Rolling stats must be in [{Rolling.VALID_STATS}]")


def _validate_windows(windows: list[int]):
    if len(windows) == 0:
        raise ValueError("Rolling windows must be non-empty")

    if any(type(window) is not int for window in windows):
        raise ValueError("Rolling windows must be integers")

    if any(window <= 0 for window in windows):
        raise ValueError("Rolling windows must be positive")

    if len(set(windows)) != len(windows):
        raise ValueError("Rolling windows must not be duplicated")


def _validate_references(references: list[str]) -> None:
    if len(references) == 0:
        raise ValueError("Ratio references must be non-empty")

    if any(reference not in Ratio.VALID_REFERENCES for reference in references):
        raise ValueError(f"Ratio references must be in [{Ratio.VALID_REFERENCES}]")
