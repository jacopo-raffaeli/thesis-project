import pandas as pd


def check_sampling_freq(
    index: pd.DatetimeIndex,
    freq: pd.Timedelta | str = "1s",
) -> None:
    """
    Assert that obj has a DatetimeIndex sampled exactly once per `freq`
    within each calendar day

    Note that the grouping per day can take a while on high frequency timeseries

    ## Args:
    * obj: A DataFrame or a Series
    * freq: The expected sampling frequency
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(index).__name__}")

    if len(index) < 2:
        raise ValueError("Expected at least two timestamps to verify frequency")

    if not index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be monotonically increasing")

    expected = pd.Timedelta(freq)

    days = index.tz_localize(None).normalize()
    deltas = index.to_series().groupby(days).diff().dropna()

    bad = deltas[deltas != expected]

    if not bad.empty:
        raise ValueError(
            f"Index is not sampled exactly at {expected} frequency. "
            f"Found {len(bad)} invalid interval(s); "
            f"first invalid interval: {bad.index[0]!s} "
            f"({bad.iloc[0]})."
        )


def check_s(s: pd.Series, check_sampling: bool = False) -> None:
    """
    Perform common checks on a Series

    ## Args:
    * s: Series
    * check_sampling: Perform sampling check (can be computationally expensive)
    """
    if not isinstance(s, pd.Series):
        raise TypeError(f"Expected a Series, got {type(s).__name__!r}")

    if s.empty:
        raise ValueError("The series is empty")

    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(s.index).__name__!r}")

    if s.index.hasnans:
        raise ValueError("Found NaNs in the index")

    if s.index.has_duplicates:
        raise ValueError("Found duplicates in the index")

    if not s.index.is_monotonic_increasing:
        raise ValueError("The index is not ordered")

    if check_sampling:
        check_sampling_freq(s.index)


def check_df(df: pd.DataFrame, check_sampling: bool = False) -> None:
    """
    Perform common checks on a DataFrame

    ## Args:
    * df: DataFrame
    * check_sampling: Perform sampling check (can be computationally expensive)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame, got {type(df).__name__!r}")

    if df.empty:
        raise ValueError("The series is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(df.index).__name__!r}")

    if df.index.hasnans:
        raise ValueError("Found NaNs in the index")

    if df.index.has_duplicates:
        raise ValueError("Found duplicates in the index")

    if not df.index.is_monotonic_increasing:
        raise ValueError("The index is not ordered")

    if check_sampling:
        check_sampling_freq(df.index)


def check_cols_in_df(lob: pd.DataFrame, columns: list[str] | str):
    if isinstance(columns, str):
        columns = [columns]

    missing = set(columns).difference(lob.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
