import datetime
import warnings
from typing import Any, Literal

import pandas as pd
from arch.unitroot import ADF, KPSS
from arch.unitroot.unitroot import kpss_crit, mackinnoncrit
from tqdm.auto import tqdm

from thesis_project import config, utils
from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.utils.io import load_filtered_parquet

Trend = Literal["c", "ct"]
Confidence = Literal["1%", "5%", "10%"]


DEFAULT_TRENDS: list[Trend] = ["c", "ct"]


def adf(
    data: pd.Series,
    *,
    lags: int,
    trend: Trend = "c",
) -> dict[str, Any]:
    """
    Run the Augmented Dickey-Fuller test.

    ## Args:
    * data: The data to test for a unit root
    * lags: The number of lags to use in the ADF regression
    * trend: The trend component to include in the test

    ## Returns:
    * results: dict of relevant ADF outputs
    """
    res = ADF(y=data, trend=trend, lags=lags)

    return {
        "nobs": res.nobs,
        "pvalue": res.pvalue,
        "stat": res.stat,
        "critical_values": res.critical_values,
    }


def kpss(
    data: pd.Series,
    *,
    lags: int,
    trend: Trend = "c",
) -> dict[str, Any]:
    """
    Run the ADF test.

    ## Args:
    * data: The data to test for a unit root
    * lags: The number of lags to use in the KPSS regression
    * trend: The trend component to include in the test

    ## Returns:
    * results: dict of relevant KPSS outputs
    """
    warnings.filterwarnings(
        "ignore",
        message="Lag selection has changed to use a data-dependent method.*",
        category=DeprecationWarning,
    )

    res = KPSS(y=data, trend=trend, lags=lags)

    return {
        "nobs": res.nobs,
        "pvalue": res.pvalue,
        "stat": res.stat,
        "critical_values": res.critical_values,
    }


def adf_critical_values(
    *,
    nobs: int,
    trend: Trend,
) -> dict[Confidence, float]:
    """
    Return ADF MacKinnon critical values for a given sample size and trend.

    ## Args:
    * nobs: Number of observation
    * trend: The trend component to include in the test

    ## Returns:
    * crit_values: three critical values, for 1%, 5% and 10% cut-offs.
    """
    values = mackinnoncrit(
        num_unit_roots=1,
        regression=trend,
        nobs=nobs,
        dist_type="adf-t",
    )

    return {
        "1%": values[0],
        "5%": values[1],
        "10%": values[2],
    }


def kpss_critical_values(
    *,
    stat: float,
    trend: Trend,
) -> dict[Confidence, float]:
    """
    Return KPSS critical values for a given statistic and trend.

    ## Args:
    * stat:The KPSS test statistic.
    * trend: The trend component to include in the test

    ## Returns:
    * crit_values: three critical values, for 1%, 5% and 10% cut-offs.
    """
    _, values = kpss_crit(stat, trend)

    return {
        "1%": values[2],
        "5%": values[1],
        "10%": values[0],
    }


def critical_values(
    test: Literal["ADF", "KPSS"],
    *,
    nobs: int,
    stat: float,
    trend: Trend,
) -> dict[Confidence, float]:
    """
    Retrieve critical values for a previously computed test result.

    ## Args:
    * nobs: Number of observation
    * trend: The trend component to include in the test

    ## Returns:
    * crit_values: three critical values, for 1%, 5% and 10% cut-offs.
    """
    if test == "ADF":
        return adf_critical_values(nobs=nobs, trend=trend)

    if test == "KPSS":
        return kpss_critical_values(stat=stat, trend=trend)

    raise ValueError(f"Unknown test: {test}")


def split_daily(s: pd.Series) -> list[tuple[datetime.date, pd.Timestamp, pd.Series]]:
    """
    Split a series into one observation window per day.

    ## Args:
    * s: Series to split

    ## Returns:
    * split: list of (date, time, series)
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a DatetimeIndex.")

    s = s.copy()
    s.index = s.index.tz_localize(None)  # type: ignore
    normalized = utils.misc.naive_dates(s.index)  # type: ignore

    result = []

    for date, group in s.groupby(normalized):
        result.append((date.date(), group.index[0], group))

    return result


def split_hourly(
    s: pd.Series,
) -> list[tuple[datetime.date, pd.Timestamp, pd.Series]]:
    """
    Split a series into non-overlapping, one hour long windows.

    ## Args:
    * s: Series to split

    ## Returns:
    * split: list of (date, time, series)
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a DatetimeIndex.")

    s = s.copy()
    s.index = s.index.tz_localize(None)  # type: ignore
    normalized = utils.misc.naive_dates(s.index)  # type: ignore

    result = []

    for date, group in s.groupby(normalized):
        min_hour = group.index.min().hour
        max_hour = group.index.max().hour

        for hour in range(min_hour, max_hour + 1):
            start = date + pd.Timedelta(hours=hour)
            end = start + pd.Timedelta(hours=1)

            window = group.between_time(
                start_time=start.time(),
                end_time=end.time(),
                inclusive="left",
            )

            if len(window):
                result.append((date.date(), start, window))

    return result


def prepare_series(
    data: pd.Series,
    *,
    frequency: str,
) -> pd.Series:
    """
    Resample a window and prepare it for unit-root testing.

    Resampling uses the last available observation in each bin.
    Missing observations are interpolated forward in time.
    Remaining leading NaNs are removed.

    ## Args:
    * data: The data to prepare
    * frequency: Pandas valid resampling frequency

    # Returns:
    * data: Resampled and cleaned Series
    """
    data = data.resample(frequency).last()

    data = data.interpolate(
        method="time",
        limit_direction="forward",
    )

    data = data.dropna()

    return data


def validate_lags(
    *,
    nobs: int,
    lags: list[int],
) -> None:
    """
    Validate that requested lags are feasible for a sample.

    ## Args:
    * nobs: Series length
    * lags: lags to use
    """
    if any(lag < 0 for lag in lags):
        raise ValueError("Lags must be non-negative.")

    if any(lag >= nobs for lag in lags):
        raise ValueError(f"Requested lag is too large for sample size nobs={nobs}: {lags}")


def _run_tests_for_lag(
    data: pd.Series,
    *,
    date: datetime.date,
    window_start: pd.Timestamp,
    frequency: str,
    lag: int,
    trends: list[Trend],
) -> list[dict[str, Any]]:
    """
    Run ADF and KPSS for one observation window and one lag.

    ## Args:
    * data: The data to test for a unit root
    * date: date the data belongs to
    * window_start: first timestamp of the time window to test
    * frequency: Pandas valid resampling frequency
    * lag: The number of lags to use in the regression
    * trend: The trend component to include in the test

    ## Returns:
    * records: list of records of the tests
    """
    records = []

    for trend in trends:
        result = adf(data, lags=lag, trend=trend)

        records.append(
            {
                "date": date,
                "window_start": window_start,
                "frequency": frequency,
                "test": "ADF",
                "trend": trend,
                "lags": lag,
                "nobs": result["nobs"],
                "stat": result["stat"],
                "pvalue": result["pvalue"],
            }
        )

        result = kpss(data, lags=lag, trend=trend)

        records.append(
            {
                "date": date,
                "window_start": window_start,
                "frequency": frequency,
                "test": "KPSS",
                "trend": trend,
                "lags": lag,
                "nobs": result["nobs"],
                "stat": result["stat"],
                "pvalue": result["pvalue"],
            }
        )

    return records


def _run_tests(
    data: pd.Series,
    *,
    date: datetime.date,
    window_start: pd.Timestamp,
    frequency: str,
    lags: list[int],
    trends: list[Trend],
) -> list[dict[str, Any]]:
    """
    Run ADF and KPSS for one observation window.

    ## Args:
    * data: The data to test for a unit root
    * date: Date the data belongs to
    * window_start: First timestamp of the time window to test
    * frequency: Pandas valid resampling frequency
    * lags: The number of lags to use in the regression
    * trends: Trend specifications to test

    ## Returns:
    * records: List of records of the tests
    """
    validate_lags(nobs=len(data), lags=lags)

    records = []

    for lag in lags:
        records.extend(
            _run_tests_for_lag(
                data,
                date=date,
                window_start=window_start,
                frequency=frequency,
                lag=lag,
                trends=trends,
            )
        )

    return records


def run_stationarity_tests(
    s: pd.Series,
    *,
    window: Literal["daily", "hourly"] = "daily",
    trends: list[Trend] = DEFAULT_TRENDS,
    freq_lags_dict: dict[str, list[int]],
) -> pd.DataFrame:
    """
    Run ADF and KPSS tests over daily or hourly windows.

    ## Args:
    *

    ## Returns:
    * records: DataFrame containing all test results.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a DatetimeIndex.")

    if window == "daily":
        windows = split_daily(s)
    elif window == "hourly":
        windows = split_hourly(s)
    else:
        raise ValueError(f"Unknown window type: {window}")

    records = []

    for frequency, lags in tqdm(freq_lags_dict.items()):
        for date, window_start, data in windows:
            prepared = prepare_series(
                data,
                frequency=frequency,
            )

            if len(prepared) < 5:
                continue

            records.extend(
                _run_tests(
                    prepared,
                    date=date,
                    window_start=window_start,
                    frequency=frequency,
                    lags=lags,
                    trends=trends,
                )
            )

    return pd.DataFrame.from_records(records)


def save(
    results: pd.DataFrame,
    *,
    ticker: config.FutTicker,
    name: str,
    window: Literal["daily", "hourly"],
) -> None:
    """
    Save each frequency/lag combination to its own parquet file.

    ## Args:
    *

    ## Returns:
    *
    """
    required = {"frequency", "lags"}

    if not required.issubset(results.columns):
        raise ValueError(f"Missing columns: {required - set(results.columns)}")

    root = config.RES_EXP_DIR / ticker / "stationarity"

    for (frequency, n_lags), group in results.groupby(
        ["frequency", "lags"],
        sort=False,
    ):
        filename = f"{name}_stationarity_{window}_freq_{frequency}_lags_{n_lags}.parquet"
        path = root / filename

        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(path, index=False)


def load(
    *,
    ticker: config.FutTicker,
    name: str,
    window: Literal["daily", "hourly"],
    frequency: str,
    n_lags: int,
) -> pd.DataFrame:
    """
    Load one frequency/lag stationarity result.

    ## Args:
    *

    ## Returns:
    *
    """
    root = config.RES_EXP_DIR / ticker / "stationarity"
    filename = f"{name}_stationarity_{window}_freq_{frequency}_lags_{n_lags}.parquet"

    return pd.read_parquet(root / filename)


def main(
    ticker: config.FutTicker,
    name: str,
    *,
    window: Literal["daily", "hourly"],
    freq_lags_dict: dict[str, list[int]],
    dates_to_exclude: list[datetime.date],
):
    s = load_filtered_parquet(
        BASE_FEATURES[name].path,
        dates_to_exclude=dates_to_exclude,
    )

    results = run_stationarity_tests(s, freq_lags_dict=freq_lags_dict)

    save(
        results,
        ticker=ticker,
        name=name,
        window=window,
    )
