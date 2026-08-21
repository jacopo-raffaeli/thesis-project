import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd

from thesis_project import config, utils
from thesis_project.profitability import common, settings


@dataclass(frozen=True)
class FixedProfitConfig:
    ticker: config.FutTicker
    ctd_contracts: int
    profits: list[float]
    max_holding_time: int | None
    analyses: list[settings.AnalysisConfig]
    min_time: datetime.time = config.STD_OPENING_TIME
    max_time: datetime.time = config.STD_CLOSING_TIME


def session_bounds(
    index: pd.DatetimeIndex,
) -> list[tuple[int, int]]:
    if len(index) == 0:
        return []

    bounds = []
    start = 0

    for i in range(1, len(index)):
        if index[i] - index[i - 1] > settings.SAMPLING_INTERVAL:
            bounds.append((start, i - 1))
            start = i

    bounds.append((start, len(index) - 1))

    return bounds


def pnl_metrics_from_path(
    path: np.ndarray,
    profit: float,
    entry_time: pd.Timestamp,
    path_times: pd.DatetimeIndex,
) -> tuple[bool, float, float]:
    valid = np.isfinite(path)

    if not valid.any():
        return False, np.nan, np.nan

    valid_path = path[valid]
    valid_times = path_times[valid]

    hits = np.flatnonzero(valid_path >= profit)

    if len(hits) == 0:
        mae = max(0.0, float(-np.min(valid_path)))
        return False, np.nan, mae

    hit_pos = hits[0]

    mae = max(
        0.0,
        float(-np.min(valid_path[: hit_pos + 1])),
    )

    time_to_profit = (valid_times[hit_pos] - entry_time).total_seconds()

    return True, time_to_profit, mae


def run_fixed_profit_single(
    data: pd.DataFrame,
    fut_contracts: pd.Series,
    profit: float,
    max_holding_time: int | None,
    ctd_contracts: int,
    price_mode: settings.PriceMode,
    volume_mode: settings.VolumeMode,
) -> pd.DataFrame:
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")

    if not data.index.is_monotonic_increasing:
        raise ValueError("data index must be sorted")

    if not data.index.equals(fut_contracts.index):
        raise ValueError("data and fut_contracts indexes must match")

    prices = common.execution_prices(data, price_mode)

    index = data.index
    n = len(index)

    entry_mask = common.execution_mask(
        data,
        fut_contracts,
        ctd_contracts,
        volume_mode,
    )

    long_allowed = entry_mask["long"].to_numpy(dtype=bool)
    short_allowed = entry_mask["short"].to_numpy(dtype=bool)

    ctd_bid = prices["ctd_bid_price"].to_numpy(dtype=float)
    ctd_ask = prices["ctd_ask_price"].to_numpy(dtype=float)
    fut_bid = prices["fut_bid_price"].to_numpy(dtype=float)
    fut_ask = prices["fut_ask_price"].to_numpy(dtype=float)
    fut_n = fut_contracts.to_numpy(dtype=float)

    ctd_scale = ctd_contracts * settings.CTD_FACE_VALUE / 100
    fut_scale = fut_n * settings.FUT_FACE_VALUE / 100

    # PnL of an open long-basis position marked at the current prices.
    long_mark = ctd_scale * ctd_bid - fut_scale * fut_ask

    # Entry cash flow of a long-basis position.
    long_entry = fut_scale * fut_bid - ctd_scale * ctd_ask

    # PnL of an open short-basis position marked at the current prices.
    short_mark = fut_scale * fut_bid - ctd_scale * ctd_ask

    # Entry cash flow of a short-basis position.
    short_entry = ctd_scale * ctd_bid - fut_scale * fut_ask

    long_profit_hit = np.zeros(n, dtype=bool)
    long_time_to_profit = np.full(n, np.nan)
    long_mae = np.full(n, np.nan)

    short_profit_hit = np.zeros(n, dtype=bool)
    short_time_to_profit = np.full(n, np.nan)
    short_mae = np.full(n, np.nan)

    for session_start, session_end in session_bounds(index):
        session_times = index[session_start : session_end + 1]
        session_length = len(session_times)

        if session_length <= 1:
            continue

        session_long_mark = long_mark[session_start : session_end + 1]
        session_long_entry = long_entry[session_start : session_end + 1]

        session_short_mark = short_mark[session_start : session_end + 1]
        session_short_entry = short_entry[session_start : session_end + 1]

        session_long_allowed = long_allowed[session_start : session_end + 1]
        session_short_allowed = short_allowed[session_start : session_end + 1]

        if max_holding_time is None:
            end_positions = np.full(
                session_length,
                session_length - 1,
                dtype=int,
            )
        else:
            holding_end = session_times + pd.Timedelta(max_holding_time, unit="s")

            end_positions = (
                np.searchsorted(
                    session_times,
                    holding_end,
                    side="right",
                )
                - 1
            )

        for i in range(session_length - 1):
            if not session_long_allowed[i] and not session_short_allowed[i]:
                continue

            end = end_positions[i]

            if end <= i:
                continue

            path_start = i + 1
            path_end = end + 1

            path_times = session_times[path_start:path_end]

            if session_long_allowed[i]:
                long_path = session_long_mark[path_start:path_end] + session_long_entry[i]

                hit, time_to_profit, mae = pnl_metrics_from_path(
                    long_path,
                    profit,
                    session_times[i],
                    path_times,
                )

                if hit:
                    position = session_start + i

                    long_profit_hit[position] = True
                    long_time_to_profit[position] = time_to_profit
                    long_mae[position] = mae

            if session_short_allowed[i]:
                short_path = session_short_mark[path_start:path_end] + session_short_entry[i]

                hit, time_to_profit, mae = pnl_metrics_from_path(
                    short_path,
                    profit,
                    session_times[i],
                    path_times,
                )

                if hit:
                    position = session_start + i

                    short_profit_hit[position] = True
                    short_time_to_profit[position] = time_to_profit
                    short_mae[position] = mae

    return pd.DataFrame(
        {
            "long_profit_hit": long_profit_hit,
            "long_time_to_profit": long_time_to_profit,
            "long_mae": long_mae,
            "short_profit_hit": short_profit_hit,
            "short_time_to_profit": short_time_to_profit,
            "short_mae": short_mae,
        },
        index=index,
    )


def run_fixed_profit(
    analysis_config: FixedProfitConfig,
) -> dict:
    data = common.load_market_data(
        ticker=analysis_config.ticker,
        min_time=analysis_config.min_time,
        max_time=analysis_config.max_time,
    )

    cf = utils.io.load_cf(analysis_config.ticker)["CF"]
    cf = common.align_cf(data, cf)

    fractional_fut_contracts = common.frac_fut_contracts(
        cf,
        analysis_config.ctd_contracts,
    )

    rounded_fut_contracts = common.round_fut_contracts(
        fractional_fut_contracts,
    )

    analyses = {}
    summaries = {}

    for analysis in analysis_config.analyses:
        match analysis.fut_contract_mode:
            case "frac":
                fut_contracts = fractional_fut_contracts

            case "round":
                fut_contracts = rounded_fut_contracts

            case _:
                raise ValueError(f"Unknown {analysis.fut_contract_mode=}")

        key = analysis.key

        analyses[key] = {}
        summaries[key] = {}

        for volume_mode in analysis.volume_modes:
            profit_analysis = {}

            for profit in analysis_config.profits:
                profit_analysis[profit] = run_fixed_profit_single(
                    data=data,
                    fut_contracts=fut_contracts,
                    profit=profit,
                    max_holding_time=analysis_config.max_holding_time,
                    ctd_contracts=analysis_config.ctd_contracts,
                    price_mode=analysis.price_mode,
                    volume_mode=volume_mode,
                )

            analyses[key][volume_mode] = profit_analysis
            summaries[key][volume_mode] = summarize_fixed_profit(
                profit_analysis,
            )

    return {
        "fractional_fut_contracts": fractional_fut_contracts,
        "rounded_fut_contracts": rounded_fut_contracts,
        "original_cf": cf,
        "effective_cf": common.compute_eff_cf(
            analysis_config.ctd_contracts,
            rounded_fut_contracts,
        ),
        "analyses": analyses,
        "summaries": summaries,
    }


def summarize_fixed_profit(
    analysis: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    results = {}

    for profit, result in analysis.items():
        for direction in ("long", "short"):
            hit = result[f"{direction}_profit_hit"]
            successful = result.loc[hit]

            time_to_profit = successful[f"{direction}_time_to_profit"].dropna()

            mae = successful[f"{direction}_mae"].dropna()

            assert isinstance(result.index, pd.DatetimeIndex)

            n_days = common.count_days(result.index)

            results[(profit, direction)] = pd.Series(
                {
                    "candidate_entries": len(result),
                    "profit_hits": int(hit.sum()),
                    "profit_hit_pct": (hit.mean() * 100 if len(hit) else np.nan),
                    "profit_hits_per_day": (hit.sum() / n_days if n_days else np.nan),
                    "mean_time_to_profit": time_to_profit.mean(),
                    "median_time_to_profit": time_to_profit.median(),
                    "max_time_to_profit": time_to_profit.max(),
                    "mean_mae": mae.mean(),
                    "median_mae": mae.median(),
                    "max_mae": mae.max(),
                }
            )

    summary = pd.DataFrame(results).T
    summary.index.names = ["profit", "direction"]

    return summary


def main():
    from thesis_project.profitability import (
        fixed_profit,
        settings,
    )

    ANALYSES = [
        settings.AnalysisConfig(
            price_mode="mid", volume_modes=("ignore", "level"), fut_contract_mode="frac"
        ),
    ]

    CONFIG = fixed_profit.FixedProfitConfig(
        ticker="fbtp",
        ctd_contracts=1,
        profits=[
            100,
        ],
        max_holding_time=None,
        analyses=ANALYSES,
    )

    fixed_profit.run_fixed_profit(CONFIG)


if __name__ == "__main__":
    main()
