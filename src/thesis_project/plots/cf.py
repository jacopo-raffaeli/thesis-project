from typing import Literal

import matplotlib.pyplot as plt

from thesis_project import config
from thesis_project.utils.io import load_cf


def cf_ts(
    ticker: config.FutTicker,
    key: Literal["ISIN", "CUSIP"] = "ISIN",
):
    df = load_cf(ticker)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        df.index,
        df["CF"],
        color="black",
        linewidth=1,
        zorder=1,
    )

    colors = plt.cm.tab20.colors  # type: ignore
    for i, k in enumerate(df[key].unique()):
        mask = df[key] == k

        ax.plot(df.index, df["CF"].where(mask), label=k, zorder=2, color=colors[i])

    ax.set_xlabel("Time")
    ax.set_ylabel("CF")
    ax.set_title("CF Timeseries")

    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        title=key,
        loc="lower right",
        ncol=2,
        fontsize=8,
        title_fontsize=9,
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    cf_ts("fbtp")
