import pandas as pd


def is_sampled_at_freq(obj: pd.DataFrame | pd.Series, freq: pd.Timedelta | str = "1s") -> None:
    """
    Assert that obj has a DatetimeIndex sampled exactly once per 'freq' (default '1s').

    ## Args:
    * obj: A DataFrame or a Series
    * freq: The frequency to check
    """
    index = obj.index

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(index).__name__}")

    if len(index) < 2:
        raise ValueError("Expected at least two timestamps to verify 1-second frequency")

    expected = pd.Timedelta(freq)
    deltas = index.to_series().diff().iloc[1:]

    if not (deltas == expected).all():
        bad = deltas[deltas != expected]
        raise ValueError(
            f"Index is not sampled exactly at {expected} frequency."
            f"Found {len(bad)} invalid interval(s); "
            f"first invalid interval: {bad.index[0]!s} "
            f"({bad.iloc[0]})."
        )
