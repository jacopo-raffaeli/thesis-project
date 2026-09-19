import matplotlib.pyplot as plt
import pandas as pd

from thesis_project import config, utils
from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.utils.io import load_filtered_parquet
from thesis_project.utils.misc import align_cf, get_dates_to_exclude

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

    return pd.DataFrame(
        {
            "ctd_mid": ctd_mid,
            "fut_mid": fut_mid,
            "ctd_spread": ctd_spread,
            "fut_spread": fut_spread,
        }
    )


def get_fut_contracts(
    ticker: config.FutTicker,
    ctd_contracts: int,
    reference: pd.DataFrame,
) -> pd.Series:
    cf = utils.io.load_cf(ticker)["CF"]

    fut_contracts = utils.misc.frac_fut_contracts(
        cf,
        ctd_contracts,
    )
    fut_contracts = utils.misc.round_fut_contracts(
        fut_contracts,
    )
    fut_contracts = align_cf(
        reference,
        fut_contracts,
    )

    return fut_contracts


def preprocess_data(
    data: pd.DataFrame,
    ctd_contracts: int,
    fut_contracts: pd.Series,
) -> pd.DataFrame:
    ctd_mid = data["ctd_mid"] * ctd_contracts * CTD_SCALE
    fut_mid = data["fut_mid"] * fut_contracts * FUT_SCALE
    basis_mid = ctd_mid - fut_mid

    ctd_spread = data["ctd_spread"] * ctd_contracts * CTD_SCALE
    fut_spread = data["fut_spread"] * fut_contracts * FUT_SCALE
    basis_spread = ctd_spread + fut_spread

    return pd.DataFrame(
        {
            "ctd_mid": ctd_mid,
            "fut_mid": fut_mid,
            "ctd_spread": ctd_spread,
            "fut_spread": fut_spread,
            "basis_mid": basis_mid,
            "basis_spread": basis_spread,
        },
        index=data.index,
    )


def analyze(
    data: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")

    grouped = data.groupby(
        pd.Grouper(freq=freq),
        sort=True,
    )

    rows = []

    for period, group in grouped:
        if group.empty:
            continue

        opening_date = group.index[0].date
        closing_date = group.index[-1].date

        min_time = group["basis_mid"].idxmin()
        max_time = group["basis_mid"].idxmax()

        # Min mid prices
        min_ctd_mid = group.loc[min_time, "ctd_mid"]
        min_fut_mid = group.loc[min_time, "fut_mid"]
        min_basis_mid = group.loc[min_time, "basis_mid"]

        # Max mid prices
        max_ctd_mid = group.loc[max_time, "ctd_mid"]
        max_fut_mid = group.loc[max_time, "fut_mid"]
        max_basis_mid = group.loc[max_time, "basis_mid"]

        # Min spread
        min_ctd_spread = group.loc[min_time, "ctd_spread"]
        min_fut_spread = group.loc[min_time, "fut_spread"]
        min_basis_spread = group.loc[min_time, "basis_spread"]

        # Max spread
        max_ctd_spread = group.loc[max_time, "ctd_spread"]
        max_fut_spread = group.loc[max_time, "fut_spread"]
        max_basis_spread = group.loc[max_time, "basis_spread"]

        if min_time < max_time:  # type: ignore
            direction = "long"
            entry_time = min_time
            exit_time = max_time
            entry_basis = min_basis_mid
            exit_basis = max_basis_mid

        if min_time > max_time:  # type: ignore
            direction = "short"
            entry_time = max_time
            exit_time = min_time
            entry_basis = max_basis_mid
            exit_basis = min_basis_mid

        gross_pnl = abs(max_basis_mid - min_basis_mid)  # type: ignore
        cost = (min_basis_spread + max_basis_spread) / 2  # type: ignore

        rows.append(
            {
                "period": period,
                "opening_date": opening_date,
                "closing_date": closing_date,
                "min_time": min_time,
                "max_time": max_time,
                "min_ctd_mid": min_ctd_mid,
                "min_fut_mid": min_fut_mid,
                "min_basis_mid": min_basis_mid,
                "max_ctd_mid": max_ctd_mid,
                "max_fut_mid": max_fut_mid,
                "max_basis_mid": max_basis_mid,
                "min_ctd_spread": min_ctd_spread,
                "min_fut_spread": min_fut_spread,
                "min_basis_spread": min_basis_spread,
                "max_ctd_spread": max_ctd_spread,
                "max_fut_spread": max_fut_spread,
                "max_basis_spread": max_basis_spread,
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
        "min_ctd_mid",
        "min_fut_mid",
        "min_basis_mid",
        "max_ctd_mid",
        "max_fut_mid",
        "max_basis_mid",
        "entry_basis",
        "exit_basis",
        "gross_pnl",
        "min_ctd_spread",
        "min_fut_spread",
        "min_basis_spread",
        "max_ctd_spread",
        "max_fut_spread",
        "max_basis_spread",
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
        result["max_basis_mid"],
        label="Max Basis",
    )
    ax.plot(
        result.index,
        result["min_basis_mid"],
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
        result["max_basis_spread"],
        label="Spread at Max Basis",
    )
    ax.plot(
        result.index,
        result["min_basis_spread"],
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
    *,
    ticker: config.FutTicker = "fbtp",
    ctd_contracts: int = 1,
    freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_data(
        ticker=ticker,
        ctd_contracts=ctd_contracts,
    )

    fut_contracts = get_fut_contracts(ticker, ctd_contracts, raw)

    data = preprocess_data(data=raw, ctd_contracts=ctd_contracts, fut_contracts=fut_contracts)

    result = analyze(
        data,
        freq=freq,
    )

    summary = make_summary(result)

    print(summary)

    plot_extrema(result)
    plot_spreads(result)

    return result, summary


if __name__ == "__main__":
    run(freq="3D")
