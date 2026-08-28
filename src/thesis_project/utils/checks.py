import pandas as pd


def is_sampled_at_freq(
    obj: pd.DataFrame | pd.Series,
    freq: pd.Timedelta | str = "1s",
) -> None:
    """
    Assert that obj has a DatetimeIndex sampled exactly once per `freq`
    within each calendar day.
    """
    index = obj.index

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(index).__name__}")

    if len(index) < 2:
        raise ValueError("Expected at least two timestamps to verify frequency")

    if not index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be monotonically increasing")

    expected = pd.Timedelta(freq)

    days = index.normalize()
    deltas = index.to_series().groupby(days).diff().dropna()

    bad = deltas[deltas != expected]

    if not bad.empty:
        raise ValueError(
            f"Index is not sampled exactly at {expected} frequency. "
            f"Found {len(bad)} invalid interval(s); "
            f"first invalid interval: {bad.index[0]!s} "
            f"({bad.iloc[0]})."
        )
