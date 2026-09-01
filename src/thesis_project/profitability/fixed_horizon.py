import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from thesis_project import config, utils
from thesis_project.profitability import common, settings
from thesis_project.utils import misc


@dataclass(frozen=True)
class FixedHorizonConfig:
    ticker: config.FutTicker
    ctd_contracts: int
    horizons: list[int]
    tolerance: int
    analyses: list[settings.AnalysisConfig]
    min_time: datetime.time = config.DEFAULT_OPENING_TIME
    max_time: datetime.time = config.DEFAULT_CLOSING_TIME


def episode_pnl(
    pnl: pd.Series,
    tolerance: int = 0,
) -> tuple[pd.Series, pd.Series]:
    profitable = pnl[pnl > 0]

    if profitable.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int)

    gaps = profitable.index.to_series().diff() > settings.SAMPLING_INTERVAL + pd.to_timedelta(
        tolerance, unit="s"
    )
    groups = gaps.fillna(False).cumsum()

    return (
        profitable.groupby(groups).mean(),
        profitable.groupby(groups).size(),
    )


def run_fixed_horizon_single(
    horizon: int,
    data: pd.DataFrame,
    fut_contracts: pd.Series,
    ctd_contracts: int,
    price_mode: config.PriceMode,
    volume_mode: config.VolumeMode,
    ctd_face_value: float = settings.CTD_FACE_VALUE,
    fut_face_value: float = settings.FUT_FACE_VALUE,
) -> pd.DataFrame:
    prices = common.execution_prices(data, price_mode)

    ctd_bid_exit = common.shift_forward(
        prices["ctd_bid_price"],
        horizon,
    )
    ctd_ask_exit = common.shift_forward(
        prices["ctd_ask_price"],
        horizon,
    )
    fut_bid_exit = common.shift_forward(
        prices["fut_bid_price"],
        horizon,
    )
    fut_ask_exit = common.shift_forward(
        prices["fut_ask_price"],
        horizon,
    )

    long_pnl = (
        ctd_contracts * ctd_face_value * (ctd_bid_exit - prices["ctd_ask_price"]) / 100
        + fut_contracts * fut_face_value * (prices["fut_bid_price"] - fut_ask_exit) / 100
    ).rename("long_pnl")

    short_pnl = (
        ctd_contracts * ctd_face_value * (prices["ctd_bid_price"] - ctd_ask_exit) / 100
        + fut_contracts * fut_face_value * (fut_bid_exit - prices["fut_ask_price"]) / 100
    ).rename("short_pnl")

    pnl = pd.DataFrame(
        {
            "long": long_pnl,
            "short": short_pnl,
        }
    )

    entry_mask = common.entry_execution_mask(
        data,
        fut_contracts,
        ctd_contracts,
        volume_mode,
    )

    exit_mask = pd.DataFrame(
        {
            "long": common.shift_forward(entry_mask["long"], horizon),
            "short": common.shift_forward(entry_mask["short"], horizon),
        },
        index=data.index,
    )

    return pnl.mask(~(entry_mask & exit_mask))


def run_fixed_horizon(
    analysis_config: FixedHorizonConfig,
):
    data = common.load_market_data(
        ticker=analysis_config.ticker,
        min_time=analysis_config.min_time,
        max_time=analysis_config.max_time,
    )

    cf = utils.io.load_cf(analysis_config.ticker)["CF"]
    cf = misc.align_cf(data, cf)

    fut_last_trading_dates = utils.io.load_fut_rollover_dates(analysis_config.ticker)
    ctd_switch_dates = utils.io.load_ctd_switch_dates(analysis_config.ticker)

    fractional_fut_contracts = misc.frac_fut_contracts(
        cf,
        analysis_config.ctd_contracts,
    )
    rounded_fut_contracts = misc.round_fut_contracts(
        fractional_fut_contracts,
    )

    analyses = {}
    summaries = {}

    for analysis in tqdm(analysis_config.analyses, desc="Analyses"):
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
            horizon_analysis = {}

            for horizon in tqdm(
                analysis_config.horizons,
                desc=f"Horizons ({volume_mode=})",
                leave=False,
            ):
                pnl = run_fixed_horizon_single(
                    horizon=horizon,
                    data=data,
                    fut_contracts=fut_contracts,
                    ctd_contracts=analysis_config.ctd_contracts,
                    price_mode=analysis.price_mode,
                    volume_mode=volume_mode,
                )

                horizon_analysis[horizon] = common.invalidate_overnight_trades(
                    pnl,
                    horizon,
                    fut_last_trading_dates,
                    ctd_switch_dates,
                )

            analyses[key][volume_mode] = horizon_analysis
            summaries[key][volume_mode] = summarize_fixed_horizon(
                horizon_analysis,
                tolerance=analysis_config.tolerance,
            )

    for key, horizon_analysis in analyses.items():
        summary = summaries[key]
        experiment = {
            "config": analysis_config,
            # "fractional_fut_contracts": fractional_fut_contracts,
            # "rounded_fut_contracts": rounded_fut_contracts,
            # "original_cf": cf,
            # "effective_cf": common.compute_eff_cf(
            #     analysis_config.ctd_contracts,
            #     rounded_fut_contracts,
            # ),
            "analyses": horizon_analysis,
            "summaries": summary,
        }

        save_experiment(experiment, analysis_config.ticker, key[0], key[1])


def summarize_fixed_horizon_single(
    pnl: pd.Series,
    tolerance: int = 0,
) -> pd.Series:
    valid = pnl.dropna()
    profitable = valid[valid > 0]

    episode_values, episode_lengths = episode_pnl(
        pnl,
        tolerance=tolerance,
    )

    assert isinstance(valid.index, pd.DatetimeIndex)
    n_days = common.count_days(valid.index) if len(valid) else 0

    assert isinstance(pnl.index, pd.DatetimeIndex)
    trading_days = common.count_days(pnl.index)

    return pd.Series(
        {
            # Relevant day counts
            "n_days": n_days,
            "trading_days": trading_days,
            # Overall PnL
            "total_pnl": valid.sum(),
            "mean_pnl": valid.mean(),
            "median_pnl": valid.median(),
            "std_pnl": valid.std(),
            "max_pnl": valid.max(),
            "min_pnl": valid.min(),
            # Profitable observations
            "profitable_observations": len(profitable),
            "profitable_observations_pct": (
                len(profitable) / len(valid) * 100 if len(valid) else np.nan
            ),
            "profitable_observations_per_day": (len(profitable) / n_days if n_days else np.nan),
            # Profitable PnL
            "profitable_total_pnl": profitable.sum(),
            "profitable_pnl_per_day": (profitable.sum() / n_days if n_days else np.nan),
            "profitable_mean_pnl": profitable.mean(),
            "profitable_median_pnl": profitable.median(),
            "profitable_std_pnl": profitable.std(),
            "profitable_max_pnl": profitable.max(),
            "profitable_min_pnl": profitable.min(),
            # Profitable episode count
            "profitable_episodes": len(episode_values),
            "profitable_episodes_pct": (
                len(episode_values) / len(profitable) * 100 if len(profitable) else np.nan
            ),
            "profitable_episodes_per_day": (len(episode_values) / n_days if n_days else np.nan),
            # Profitable episode PnL
            "episode_total_pnl": episode_values.sum(),
            "episode_mean_pnl": episode_values.mean(),
            "episode_median_pnl": episode_values.median(),
            "episode_std_pnl": episode_values.std(),
            "episode_max_pnl": episode_values.max(),
            "episode_min_pnl": episode_values.min(),
            # Profitable episode length
            "mean_episode_length": (episode_lengths.mean() if len(episode_lengths) else np.nan),
            "median_episode_length": (episode_lengths.median() if len(episode_lengths) else np.nan),
            "max_episode_length": (episode_lengths.max() if len(episode_lengths) else np.nan),
            "min_episode_length": (episode_lengths.min() if len(episode_lengths) else np.nan),
        }
    )


def summarize_fixed_horizon(
    analysis: dict[int, pd.DataFrame],
    tolerance: int = 0,
) -> pd.DataFrame:
    results = {}

    for horizon, pnl in analysis.items():
        for direction in ("long", "short"):
            results[(horizon, direction)] = summarize_fixed_horizon_single(
                pnl[direction],
                tolerance=tolerance,
            )

    summary = pd.DataFrame(results).T
    print(summary.index.names)
    summary.index.names = ["horizon", "direction"]

    return summary


def save_experiment(
    experiment,
    ticker: config.FutTicker,
    price: config.PriceMode,
    contract: config.ContractMode,
):
    path = config.RES_EXP_DIR / ticker / "profitability" / "fixed_horizon"
    path.mkdir(parents=True, exist_ok=True)

    analyses = experiment["analyses"].items()
    for volume_mode, horizon_analyses in analyses:
        for horizon, analysis in horizon_analyses.items():
            # Save analysis
            # filename = f"price_{price}_contract_{contract}_volume_{volume_mode}_horizon_{horizon}_analysis.parquet"
            # analysis.to_parquet(path / filename)
            # Save summary
            summary = experiment["summaries"][volume_mode].xs(
                horizon, level="horizon", drop_level=False
            )
            filename = f"price_{price}_contract_{contract}_volume_{volume_mode}_horizon_{horizon}_summary.parquet"
            summary.to_parquet(path / filename)


def load_experiment(
    ticker: config.FutTicker,
    price: config.PriceMode,
    contract: config.ContractMode,
    volume: config.VolumeMode,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    path = config.RES_EXP_DIR / ticker / "profitability" / "fixed_horizon"

    # filename = (
    #     f"price_{price}_contract_{contract}_volume_{volume}_horizon_{horizon}_analysis.parquet"
    # )
    # analysis = pd.read_parquet(path / filename)

    filename = (
        f"price_{price}_contract_{contract}_volume_{volume}_horizon_{horizon}_summary.parquet"
    )
    summary = pd.read_parquet(path / filename)

    return summary
