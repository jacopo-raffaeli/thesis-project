import datetime
import pickle as pkl
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from thesis_project import config, utils
from thesis_project.profitability import common, settings


@dataclass(frozen=True)
class FixedProfitConfig:
    ticker: config.FutTicker
    ctd_contracts: int
    profits: list[float]
    max_holding_time: int | None
    n_jobs: int
    analyses: list[settings.AnalysisConfig]
    min_time: datetime.time = config.STD_OPENING_TIME
    max_time: datetime.time = config.STD_CLOSING_TIME


def session_bounds(
    index: pd.DatetimeIndex,
) -> list[tuple[int, int]]:
    if len(index) == 0:
        return []

    gaps = index[1:] - index[:-1]
    session_starts = np.flatnonzero(gaps > settings.SAMPLING_INTERVAL) + 1

    starts = np.r_[0, session_starts]
    ends = np.r_[session_starts - 1, len(index) - 1]

    return list(zip(starts, ends))


def pnl_metrics_from_path(
    path: np.ndarray,
    exit_allowed: np.ndarray,
    profit: float,
    entry_time: pd.Timestamp,
    path_times: pd.DatetimeIndex,
) -> tuple[bool, float, float]:
    valid = np.isfinite(path)

    if not valid.any():
        return False, np.nan, np.nan

    executable = valid & exit_allowed

    hits = np.flatnonzero(executable & (path >= profit))

    if len(hits) == 0:
        mae = max(0.0, float(-np.min(path[valid])))
        return False, np.nan, mae

    hit_pos = hits[0]

    mae = max(
        0.0,
        float(-np.min(path[valid][: hit_pos + 1])),
    )

    time_to_profit = (path_times[hit_pos] - entry_time).total_seconds()

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

    entry_mask = common.entry_execution_mask(
        data,
        fut_contracts,
        ctd_contracts,
        volume_mode,
    )

    exit_mask = common.exit_execution_mask(
        data,
        fut_contracts,
        ctd_contracts,
        volume_mode,
    )

    long_entry_allowed = entry_mask["long"].to_numpy(dtype=bool)
    short_entry_allowed = entry_mask["short"].to_numpy(dtype=bool)

    long_exit_allowed = exit_mask["long"].to_numpy(dtype=bool)
    short_exit_allowed = exit_mask["short"].to_numpy(dtype=bool)

    if len(long_exit_allowed) != n or len(short_exit_allowed) != n:
        raise RuntimeError(
            f"Exit mask length mismatch: "
            f"{len(long_exit_allowed)=}, "
            f"{len(short_exit_allowed)=}, "
            f"{n=}"
        )

    ctd_bid = prices["ctd_bid_price"].to_numpy(dtype=float)
    ctd_ask = prices["ctd_ask_price"].to_numpy(dtype=float)
    fut_bid = prices["fut_bid_price"].to_numpy(dtype=float)
    fut_ask = prices["fut_ask_price"].to_numpy(dtype=float)
    fut_n = fut_contracts.to_numpy(dtype=float)

    ctd_scale = ctd_contracts * settings.CTD_FACE_VALUE / 100
    fut_scale = fut_n * settings.FUT_FACE_VALUE / 100

    # PnL of an open long-basis position marked at the current prices
    long_mark = ctd_scale * ctd_bid - fut_scale * fut_ask

    # Entry cash flow of a long-basis position
    long_entry = fut_scale * fut_bid - ctd_scale * ctd_ask

    # PnL of an open short-basis position marked at the current prices
    short_mark = fut_scale * fut_bid - ctd_scale * ctd_ask

    # Entry cash flow of a short-basis position
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

        session_long_entry_allowed = long_entry_allowed[session_start : session_end + 1]
        session_short_entry_allowed = short_entry_allowed[session_start : session_end + 1]

        session_long_exit_allowed = long_exit_allowed[session_start : session_end + 1]
        session_short_exit_allowed = short_exit_allowed[session_start : session_end + 1]

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
            if not session_long_entry_allowed[i] and not session_short_entry_allowed[i]:
                continue

            end = end_positions[i]

            if end <= i:
                continue

            path_start = i + 1
            path_end = end + 1

            path_times = session_times[path_start:path_end]

            if session_long_entry_allowed[i]:
                long_path = session_long_mark[path_start:path_end] + session_long_entry[i]

                long_exit_allowed_path = session_long_exit_allowed[path_start:path_end]

                if len(long_path) != len(long_exit_allowed_path):
                    raise RuntimeError(
                        "Long path/exit mask length mismatch: "
                        f"{len(long_path)=}, "
                        f"{len(long_exit_allowed_path)=}, "
                        f"{session_start=}, "
                        f"{session_end=}, "
                        f"{session_length=}, "
                        f"{i=}, "
                        f"{end=}, "
                        f"{path_start=}, "
                        f"{path_end=}"
                    )

                hit, time_to_profit, mae = pnl_metrics_from_path(
                    long_path,
                    long_exit_allowed_path,
                    profit,
                    session_times[i],
                    path_times,
                )

                if hit:
                    position = session_start + i
                    long_profit_hit[position] = True
                    long_time_to_profit[position] = time_to_profit
                    long_mae[position] = mae

            if session_short_entry_allowed[i]:
                short_path = session_short_mark[path_start:path_end] + session_short_entry[i]

                short_exit_allowed_path = session_short_exit_allowed[path_start:path_end]

                if len(short_path) != len(short_exit_allowed_path):
                    raise RuntimeError(
                        "Short path/exit mask length mismatch: "
                        f"{len(short_path)=}, "
                        f"{len(short_exit_allowed_path)=}, "
                        f"{session_start=}, "
                        f"{session_end=}, "
                        f"{session_length=}, "
                        f"{i=}, "
                        f"{end=}, "
                        f"{path_start=}, "
                        f"{path_end=}"
                    )

                hit, time_to_profit, mae = pnl_metrics_from_path(
                    short_path,
                    short_exit_allowed_path,
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


def run_analysis(
    analysis: settings.AnalysisConfig,
    analysis_config: FixedProfitConfig,
    data: pd.DataFrame,
    fractional_fut_contracts: pd.Series,
    rounded_fut_contracts: pd.Series,
) -> tuple[tuple[settings.PriceMode, settings.FutContractMode], dict, dict]:
    match analysis.fut_contract_mode:
        case "frac":
            fut_contracts = fractional_fut_contracts
        case "round":
            fut_contracts = rounded_fut_contracts
        case _:
            raise ValueError(f"Unknown {analysis.fut_contract_mode=}")

    profit_analysis_by_volume = {}
    summaries_by_volume = {}

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

        profit_analysis_by_volume[volume_mode] = profit_analysis
        summaries_by_volume[volume_mode] = summarize_fixed_profit(
            profit_analysis,
        )

    return (
        analysis.key,
        profit_analysis_by_volume,
        summaries_by_volume,
    )


def run_fixed_profit(
    analysis_config: FixedProfitConfig,
):
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

    with tqdm_joblib(
        tqdm(
            total=len(analysis_config.analyses),
            desc="Analyses",
        )
    ):
        results = Parallel(
            n_jobs=analysis_config.n_jobs,
            backend="loky",
        )(
            delayed(run_analysis)(
                analysis,
                analysis_config,
                data,
                fractional_fut_contracts,
                rounded_fut_contracts,
            )
            for analysis in analysis_config.analyses
        )

    for key, profit_analysis, summary in results:  # type: ignore
        experiment = {
            "config": analysis_config,
            # "fractional_fut_contracts": fractional_fut_contracts,
            # "rounded_fut_contracts": rounded_fut_contracts,
            # "original_cf": cf,
            # "effective_cf": common.compute_eff_cf(
            #     analysis_config.ctd_contracts,
            #     rounded_fut_contracts,
            # ),
            "analyses": profit_analysis,
            "summaries": summary,
        }

        save_experiment(experiment, analysis_config.ticker, key[0], key[1])


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


def save_experiment(
    experiment,
    ticker: config.FutTicker,
    price: settings.PriceMode,
    contract: settings.FutContractMode,
):
    path = config.RES_EXP_DIR / ticker / "profitability" / "fixed_profit"
    path.mkdir(parents=True, exist_ok=True)

    analyses = experiment["analyses"].items()
    for volume_mode, profit_analyses in analyses:
        for profit_target, analysis in profit_analyses.items():
            # Save analysis
            filename = f"price_{price}_contract_{contract}_volume_{volume_mode}_profit_{profit_target}_analysis.parquet"
            analysis.to_parquet(path / filename)
            # Save summary
            summary = experiment["summaries"][volume_mode].xs(
                profit_target, level="profit", drop_level=False
            )
            filename = f"price_{price}_contract_{contract}_volume_{volume_mode}_profit_{profit_target}_summary.parquet"
            summary.to_parquet(path / filename)


def load_experiment(
    ticker: config.FutTicker, price: settings.PriceMode, contract: settings.FutContractMode
):
    path = (
        config.RES_EXP_DIR
        / ticker
        / "profitability"
        / "fixed_profit"
        / f"{price}_price_{contract}_contract.pkl"
    )

    with path.open("rb") as f:
        return pkl.load(f)
