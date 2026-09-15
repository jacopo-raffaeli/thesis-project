import datetime
from typing import Iterable

import matplotlib.pyplot as plt

from thesis_project import config, utils
from thesis_project.dataset.data import BASE_FEATURES

# TODO:
# Define a proper label for y-axis when price/100 is showed

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
            data = data.resample("5min").last()

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
):
    fig, ax = plt.subplots(figsize=(8, 4))

    name = f"{role}_spread"
    path = BASE_FEATURES[name].path
    data = utils.io.load_filtered_parquet(
        path,
        time_window=window,
        dates_to_include=dates,
    )
    data = data.resample("30min").last()

    ax.plot(data, label=name.replace("_", " ").title(), linewidth=0.7, color="black")

    ax.set_xlabel("Time")

    ax.set_ylabel("Price")

    ax.set_title(f"{ticker.upper()} {role.upper()} Spread")

    ax.spines[["top", "right"]].set_visible(False)

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


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

    size_ts(ticker, "ctd", window=window, dates=dates)
    size_ts(ticker, "fut", window=window, dates=dates)
