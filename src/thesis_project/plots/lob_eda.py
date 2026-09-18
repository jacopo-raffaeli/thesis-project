import datetime
from typing import Iterable, get_args

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from thesis_project import config, utils
from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.utils.misc import naive_dates

# TODO:
# Define a proper label for y-axis when price/100 is shown

# TODO:
# Evaluate if and what pieces of code are worth to be resued in the script

# TODO:
# Define a 'generic' signature for the functions:
# - When to plot a single role or both (ctd, fut)
# - An easy way to pass typical auxiliar parameters like:
#   - resample freq
#   - time window
#   - dates


# ==============================================================================
# Prices
# ==============================================================================


def price_ts(
    ticker: config.FutTicker,
    role: config.AssetRole,
    levels: Iterable[int] = (1,),
    sides: Iterable[config.LobSide] = ("bid", "ask"),
    window: tuple[datetime.time, datetime.time] | None = None,
    dates: list[datetime.date] | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    for level in levels:
        for side in sides:
            name = f"{role}_{side}_price_{level}"
            path = BASE_FEATURES[name].path
            data = utils.io.load_filtered_parquet(
                path,
                time_window=window,
                dates_to_include=dates,
            )
            data = data.resample("5min").last()

            ax.plot(data, label=name.replace("_", " ").title(), linewidth=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker.upper()} {role.upper()} Prices")

    ax.spines[["top", "right"]].set_visible(False)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def mid_price_ts(
    ticker: config.FutTicker,
    role: config.AssetRole,
    window: tuple[datetime.time, datetime.time] | None = None,
    dates: list[datetime.date] | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    name = f"{role}_mid_price"
    path = BASE_FEATURES[name].path
    data = utils.io.load_filtered_parquet(
        path,
        time_window=window,
        dates_to_include=dates,
    )
    data = data.resample("5min").last()

    ax.plot(data, label=name.replace("_", " ").title(), linewidth=0.7, color="black")

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker.upper()} {role.upper()} Mid Price")

    ax.spines[["top", "right"]].set_visible(False)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# Sizes
# ==============================================================================


def size_ts(
    ticker: config.FutTicker,
    role: config.AssetRole,
    levels: Iterable[int] = (1,),
    sides: Iterable[config.LobSide] = ("bid", "ask"),
    window: tuple[datetime.time, datetime.time] | None = None,
    dates: list[datetime.date] | None = None,
    resample_freq: str = "1h",
):
    fig, ax = plt.subplots(figsize=(8, 4))

    for level in levels:
        for side in sides:
            name = f"{role}_{side}_size_{level}"
            path = BASE_FEATURES[name].path
            data = utils.io.load_filtered_parquet(
                path,
                time_window=window,
                dates_to_include=dates,
            )
            data = data.resample(resample_freq).last()

            ax.plot(data, label=name.replace("_", " ").title(), linewidth=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel("Size")
    ax.set_title(f"{ticker.upper()} {role.upper()} Prices")

    ax.spines[["top", "right"]].set_visible(False)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# Spreads
# ==============================================================================


def spread_ts(
    ticker: config.FutTicker,
    role: config.AssetRole,
    window: tuple[datetime.time, datetime.time] | None = None,
    dates: list[datetime.date] | None = None,
    resample_freq: str = "1h",
):
    fig, ax = plt.subplots(figsize=(8, 4))

    name = f"{role}_spread"
    path = BASE_FEATURES[name].path
    data = utils.io.load_filtered_parquet(
        path,
        time_window=window,
        dates_to_include=dates,
    )
    data = data.resample(resample_freq).last()

    ax.plot(data, label=name.replace("_", " ").title(), linewidth=0.7, color="black")

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker.upper()} {role.upper()} Spread")

    ax.spines[["top", "right"]].set_visible(False)

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def spread_ts_mean(
    ticker: config.FutTicker,
    role: config.AssetRole,
    window: tuple[datetime.time, datetime.time] | None = None,
    dates: list[datetime.date] | None = None,
    resample_freq: str = "1h",
):
    fig, ax = plt.subplots(figsize=(8, 4))

    name = f"{role}_spread"
    path = BASE_FEATURES[name].path

    data = utils.io.load_filtered_parquet(
        path,
        time_window=window,
        dates_to_include=dates,
    )

    assert isinstance(data.index, pd.DatetimeIndex)
    normalized = naive_dates(data.index)
    data = data.groupby(normalized).resample(resample_freq).last().droplevel(0)

    assert isinstance(data.index, pd.DatetimeIndex)
    data.index = data.index.time
    grouped = data.groupby(data.index)

    mean = grouped.mean()
    std = grouped.std()

    x = pd.to_datetime(
        [t.strftime("%H:%M:%S") for t in mean.index],
        format="%H:%M:%S",
    )

    ax.plot(
        x,
        mean.values,  # type: ignore
        label=name.replace("_", " ").title(),
        color="tab:blue",
    )

    ax.fill_between(
        x,
        mean.values - std.values,  # type: ignore
        mean.values + std.values,  # type: ignore
        alpha=0.4,
    )

    # x-axis
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    # y-axis
    ax.set_ylabel("Price")

    # Figure
    ax.set_title(f"{ticker.upper()} {role.upper()} Spread")
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def spread_hist(
    ticker: config.FutTicker,
    role: config.AssetRole,
    window: tuple[datetime.time, datetime.time] | None = None,
    dates: list[datetime.date] | None = None,
    resample_freq: str = "1h",
):
    fig, ax = plt.subplots(figsize=(8, 4))

    name = f"{role}_spread"
    path = BASE_FEATURES[name].path
    data = utils.io.load_filtered_parquet(
        path,
        time_window=window,
        dates_to_include=dates,
    )
    data = data.resample(resample_freq).last()

    ax.hist(
        data,
        label=name.replace("_", " ").title(),
        bins=100,
    )

    ax.set_xlabel("Price")
    ax.set_ylabel("Count")
    ax.set_title(f"{ticker.upper()} {role.upper()} Spread")

    ax.spines[["top", "right"]].set_visible(False)

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def spread_bp(
    ticker: config.FutTicker,
    role: config.AssetRole,
    window: tuple[datetime.time, datetime.time] | None = None,
    dates: list[datetime.date] | None = None,
    resample_freq: str = "1h",
):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Load data
    name = f"{role}_spread"
    path = BASE_FEATURES[name].path
    data = utils.io.load_filtered_parquet(
        path,
        time_window=window,
        dates_to_include=dates,
    )

    assert isinstance(data.index, pd.DatetimeIndex)
    bucket = data.index.tz_localize(None).floor(resample_freq).time
    grouped = data.groupby(bucket)

    times = list(grouped.groups.keys())
    values = [group.to_numpy() for _, group in grouped]

    ax.boxplot(
        values,
        positions=range(len(values)),
        widths=0.6,
    )

    # x-axis
    ax.set_xlabel("Time")
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(
        [t.strftime("%H:%M:%S") for t in times],
        rotation=45,
    )

    # y-axis
    ax.set_ylabel("Price")

    # Figure
    ax.set_title(f"{ticker.upper()} {role.upper()} Spread")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    # LOB config
    ticker: config.FutTicker = "fbtp"
    levels = (1, 2)

    # Time settings
    min_time = datetime.time(9, 0, 0)
    max_time = datetime.time(17, 0, 0)
    window = (min_time, max_time)

    # Dates settings
    min_date = datetime.date(2022, 8, 1)
    max_date = datetime.date(2022, 8, 1)
    dates = [min_date + datetime.timedelta(days=i) for i in range((max_date - min_date).days + 1)]

    for role in get_args(config.AssetRole):
        spread_bp(ticker, role, window=None, dates=None, resample_freq="1h")
