"""Trade opportunity analysis"""

import pandas as pd
from tqdm.auto import tqdm


def trade_opportunity_single(
    prices: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.is_monotonic_increasing
    assert prices.index.is_unique
    assert {"bid", "ask"} <= set(prices.columns)

    target_index = prices.index + pd.to_timedelta(horizon, unit="s")
    future = prices.reindex(target_index)
    future.index = prices.index

    result = pd.DataFrame(index=prices.index)
    result["long"] = future["bid"] - prices["ask"]
    result["short"] = prices["bid"] - future["ask"]

    return result


def trade_opportunity(
    bid: pd.Series,
    ask: pd.Series,
    horizons: list[int],
) -> dict[int, pd.DataFrame]:
    assert isinstance(bid.index, pd.DatetimeIndex)
    assert isinstance(ask.index, pd.DatetimeIndex)
    assert bid.index.equals(ask.index)

    prices = pd.DataFrame(
        {
            "bid": bid,
            "ask": ask,
        }
    )

    return {horizon: trade_opportunity_single(prices, horizon) for horizon in tqdm(horizons)}


def summary_pnl(pnl: pd.Series) -> pd.Series:
    profitable = pnl > 0

    return pd.Series(
        {
            "mean pnl": pnl.mean(),
            "median pnl": pnl.median(),
            "std pnl": pnl.std(),
            "max pnl": pnl.max(),
            "min pnl": pnl.min(),
            "profitable trades": profitable.sum(),
            "profitable trades (%)": profitable.mean() * 100,
            "profitable trades mean pnl": pnl[profitable].mean(),
            "profitable trades median pnl": pnl[profitable].median(),
            "profitable trades std pnl": pnl[profitable].std(),
            "profitable trades max pnl": pnl[profitable].max(),
            "profitable trades min pnl": pnl[profitable].min(),
        }
    )


def summary_trade_opportunities(
    analysis: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    results = {}

    for horizon, pnl in analysis.items():
        for direction in ("long", "short"):
            results[(horizon, direction)] = summary_pnl(pnl[direction])

    summary = pd.DataFrame(results).T
    summary.index.names = ["horizon", "direction"]

    return summary
