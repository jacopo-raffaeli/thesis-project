from typing import get_args

import matplotlib.pyplot as plt

from thesis_project import config, dataset

# TODO: Add multiple plots per figure for levels, sides, columns
# TODO: Address a proper xticks if not perc
# TODO: Add timestamps distribution
# TODO: Add gaps distribution
# TODO: Add lobs/dates retrivial for given condition


N_NAN_TO_TITLE: dict[str, str] = {
    "n_nan": "number of NaNs",
    "n_nan_perc": "percent of NaNs",
    "n_rows_all_nan": "number of rows with all NaNs",
    "n_rows_all_nan_perc": "percent of rows with all NaNs",
    "n_rows_any_nan": "number of rows with any NaNs",
    "n_rows_any_nan_perc": "percent of rows with any NaNs",
}


def _is_perc(key: str) -> bool:
    return key.endswith("_perc")


def _nans_generic_hist(data: list[int] | list[float], key: str, title: str):
    config.default_plt()
    fig, ax = plt.subplots(figsize=(8, 4))

    if _is_perc(key):
        bins = range(0, 101, 1)
        ax.set_xlabel("Percent of NaNs")
        ax.set_xticks(range(0, 105, 5))
        ax.set_xticks(range(0, 101, 1), minor=True)
    else:
        bins = "fd"
        ax.set_xlabel("Number of NaNs")

    ax.hist(
        data,
        bins=bins,
        align="left",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylabel("Number of dates")

    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()


def nans_hist_lob(collection: dataset.lob.LobReportCollector, key: str):
    data = []
    for _, report in collection.reports.items():
        data.append(report.integrity["nans"]["lob"][key])

    title = f"Distribution of the {N_NAN_TO_TITLE[key]} in {collection.ticker.upper()} {collection.role.upper()} LOB"
    _nans_generic_hist(data, key, title)


def nans_hist_level(collection: dataset.lob.LobReportCollector, key: str, level: int):
    data = []
    for _, report in collection.reports.items():
        try:
            data.append(report.integrity["nans"]["levels"][level][key])
        except KeyError:
            continue

    if not data:
        return

    title = (
        f"Distribution of the {N_NAN_TO_TITLE[key]} in {collection.ticker.upper()} {collection.role.upper()} LOB"
        f"\nLevel {level!r}"
    )
    _nans_generic_hist(data, key, title)


def nans_hist_side(
    collection: dataset.lob.LobReportCollector, key: str, level: int, side: config.LobSide
):
    data = []
    for _, report in collection.reports.items():
        try:
            data.append(report.integrity["nans"]["sides"][(level, side)][key])
        except KeyError:
            continue

    if not data:
        return

    title = (
        f"Distribution of the {N_NAN_TO_TITLE[key]} in {collection.ticker.upper()} {collection.role.upper()} LOB"
        f"\nLevel {level!r} Side {side.capitalize()!r}"
    )
    _nans_generic_hist(data, key, title)


def nans_hist_column(
    collection: dataset.lob.LobReportCollector,
    key: str,
    level: int,
    side: config.LobSide,
    column: config.LobColumn,
):
    col = config.LOB.get_column(level=level, side=side, column_type=column)
    data = []

    if "perc" not in key:
        temp_key = "n_nan"
    else:
        temp_key = "n_nan_perc"

    for _, report in collection.reports.items():
        try:
            data.append(report.integrity["nans"]["columns"][col][temp_key])
        except KeyError:
            continue

    if not data:
        return

    title = (
        f"Distribution of the {N_NAN_TO_TITLE[key]} in {collection.ticker.upper()} {collection.role.upper()} LOB"
        f"\nLevel {level!r} Side {side.capitalize()!r} Column {column.capitalize()!r}"
    )
    _nans_generic_hist(data, key, title)


def nans_hist_per_role(collection: dataset.lob.LobReportCollector, key: str):
    nans_hist_lob(collection, key)

    for level in config.LOB.levels:
        nans_hist_level(collection, key, level)

    for level in config.LOB.levels:
        for side in get_args(config.LobSide):
            nans_hist_side(collection, key, level, side)

    for level in config.LOB.levels:
        for side in get_args(config.LobSide):
            for column in get_args(config.LobColumn):
                nans_hist_column(collection, key, level, side, column)


def nans_hist_per_ticker(
    collections: dict[config.AssetRole, dataset.lob.LobReportCollector], key: str
):
    if key not in N_NAN_TO_TITLE:
        raise ValueError(f"Unknown {key=}")

    for collection in collections.values():
        nans_hist_per_role(collection, key)
