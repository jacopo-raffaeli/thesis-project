import datetime
from typing import Iterable

import matplotlib.pyplot as plt

from thesis_project import config, utils
from thesis_project.dataset.data import BASE_FEATURES


def plot_price(
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


def plot_mid_price(
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


if __name__ == "__main__":
    TICKER: config.FutTicker = "fbtp"
    levels = (1, 2)
    window = (datetime.time(9, 0, 0), datetime.time(17, 0, 0))
    dates = [datetime.date(2023, 4, 3)]

    plot_price(TICKER, "ctd", levels=levels, dates=dates, window=window)
    plot_price(TICKER, "fut", levels=levels, dates=dates, window=window)
    plot_mid_price(TICKER, "ctd", window=window, dates=dates)
    plot_mid_price(TICKER, "fut", window=window, dates=dates)
