from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from thesis_project import config, utils
from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.utils.io import load_filtered_parquet
from thesis_project.utils.misc import align_cf, get_dates_to_exclude

Frequency = Literal["1D", "1W"]

# Analysis parameters
MIN_TIME = config.DEFAULT_OPENING_TIME
MAX_TIME = config.DEFAULT_CLOSING_TIME

EXCLUDED_DATE_OFFSETS = config.DEFAULT_EXCLUDED_DATES

CTD_SCALE = config.BTP.contract_size / 100
FUT_SCALE = config.FBTP.contract_size / 100


def load_data(
    ticker: config.FutTicker,
    ctd_contracts: int,
) -> pd.DataFrame:
    time_window = (MIN_TIME, MAX_TIME)

    excluded = get_dates_to_exclude(
        ticker,
        EXCLUDED_DATE_OFFSETS,
    )

    ctd_mid = load_filtered_parquet(
        BASE_FEATURES["ctd_mid_price"].path,
        time_window=time_window,
        dates_to_exclude=excluded,
    )

    fut_mid = load_filtered_parquet(
        BASE_FEATURES["fut_mid_price"].path,
        time_window=time_window,
        dates_to_exclude=excluded,
    )

    ctd_spread = load_filtered_parquet(
        BASE_FEATURES["ctd_spread"].path,
        time_window=time_window,
        dates_to_exclude=excluded,
    )

    fut_spread = load_filtered_parquet(
        BASE_FEATURES["fut_spread"].path,
        time_window=time_window,
        dates_to_exclude=excluded,
    )

    assert ctd_mid.index.equals(fut_mid.index)
    assert ctd_mid.index.equals(ctd_spread.index)
    assert ctd_mid.index.equals(fut_spread.index)

    cf = utils.io.load_cf(ticker)["CF"]

    fut_contracts = utils.misc.frac_fut_contracts(
        cf,
        ctd_contracts,
    )
    fut_contracts = utils.misc.round_fut_contracts(
        fut_contracts,
    )
    fut_contracts = align_cf(
        fut_mid.to_frame(),
        fut_contracts,
    )

    return pd.DataFrame(
        {
            "ctd_mid": ctd_mid,
            "fut_mid": fut_mid,
            "ctd_spread": ctd_spread,
            "fut_spread": fut_spread,
            "ctd_contracts": ctd_contracts,
            "fut_contracts": fut_contracts,
        }
    )


def preprocess_data(
    data: pd.DataFrame,
    ctd_contracts: int,
) -> pd.DataFrame:
    ctd_side = data["ctd_mid"] * ctd_contracts * CTD_SCALE

    fut_side = data["fut_mid"] * data["fut_contracts"] * FUT_SCALE

    basis = ctd_side - fut_side

    ctd_spread = data["ctd_spread"] * ctd_contracts * CTD_SCALE

    fut_spread = data["fut_spread"] * data["fut_contracts"] * FUT_SCALE

    spread = ctd_spread + fut_spread

    return pd.DataFrame(
        {
            "basis": basis,
            "spread": spread,
        },
        index=data.index,
    )


def analyze(
    data: pd.DataFrame,
    frequency: Frequency,
) -> pd.DataFrame:
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")

    grouped = data.groupby(
        pd.Grouper(freq=frequency),
        sort=True,
    )

    rows = []

    for period, group in grouped:
        if group.empty:
            continue

        min_time = group["basis"].idxmin()
        max_time = group["basis"].idxmax()

        min_basis = group.loc[min_time, "basis"]
        max_basis = group.loc[max_time, "basis"]

        min_spread = group.loc[min_time, "spread"]
        max_spread = group.loc[max_time, "spread"]

        if min_time < max_time:
            direction = "long"
            entry_time = min_time
            exit_time = max_time
            entry_basis = min_basis
            exit_basis = max_basis

        elif max_time < min_time:
            direction = "short"
            entry_time = max_time
            exit_time = min_time
            entry_basis = max_basis
            exit_basis = min_basis

        else:
            continue

        gross_pnl = abs(max_basis - min_basis)  # type: ignore

        cost = (
            min_spread / 2  # type: ignore
            + max_spread / 2  # type: ignore
        )

        rows.append(
            {
                "period": period,
                "min_time": min_time,
                "max_time": max_time,
                "min_basis": min_basis,
                "max_basis": max_basis,
                "min_spread": min_spread,
                "max_spread": max_spread,
                "direction": direction,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_basis": entry_basis,
                "exit_basis": exit_basis,
                "gross_pnl": gross_pnl,
                "cost": cost,
                "net_pnl": gross_pnl - cost,  # type: ignore
                "holding_time": exit_time - entry_time,  # type: ignore
            }
        )

    result = pd.DataFrame(rows).set_index("period")

    monetary_columns = [
        "min_basis",
        "max_basis",
        "entry_basis",
        "exit_basis",
        "gross_pnl",
        "min_spread",
        "max_spread",
        "cost",
        "net_pnl",
    ]

    result[monetary_columns] = result[monetary_columns].round(2)

    return result


def make_summary(
    result: pd.DataFrame,
) -> pd.DataFrame:
    profitable = result["net_pnl"] > 0

    rows = {
        "Overall": result,
        "Net PnL > 0": result.loc[profitable],
    }

    summaries = {}

    for name, df in rows.items():
        summaries[name] = {
            "N. periods": len(df),
            "Total Gross PnL": df["gross_pnl"].sum(),
            "Total Cost": df["cost"].sum(),
            "Total Net PnL": df["net_pnl"].sum(),
            "Mean Gross PnL": df["gross_pnl"].mean(),
            "Mean Cost": df["cost"].mean(),
            "Mean Net PnL": df["net_pnl"].mean(),
            "N. Long Trades": (df["direction"] == "long").sum(),
            "N. Short Trades": (df["direction"] == "short").sum(),
            "Long Trades (%)": ((df["direction"] == "long").mean() * 100),
            "Short Trades (%)": ((df["direction"] == "short").mean() * 100),
        }

    return pd.DataFrame(summaries).T


def plot_extrema(result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        result.index,
        result["max_basis"],
        label="Max Basis",
    )
    ax.plot(
        result.index,
        result["min_basis"],
        label="Min Basis",
    )

    ax.set(
        xlabel="Period",
        ylabel="€",
        title="Basis Extrema",
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_spreads(result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        result.index,
        result["max_spread"],
        label="Spread at Max Basis",
    )
    ax.plot(
        result.index,
        result["min_spread"],
        label="Spread at Min Basis",
    )

    ax.set(
        xlabel="Period",
        ylabel="€",
        title="Spreads at Basis Extrema",
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()

    fig.tight_layout()
    plt.show()


def run(
    ticker: config.FutTicker,
    ctd_contracts: int,
    frequency: Frequency,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_data(
        ticker=ticker,
        ctd_contracts=ctd_contracts,
    )

    data = preprocess_data(
        raw,
        ctd_contracts=ctd_contracts,
    )

    result = analyze(
        data,
        frequency=frequency,
    )

    summary = make_summary(result)

    print(summary)

    plot_extrema(result)
    plot_spreads(result)

    return result, summary
