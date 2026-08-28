from typing import get_args

import matplotlib.pyplot as plt

from thesis_project import config, dataset

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


def _collect_lob_data(
    collection: dataset.lob.LobReportCollector,
    key: str,
) -> list[int] | list[float]:
    return [report.integrity["nans"]["lob"][key] for _, report in collection.reports.items()]


def _collect_level_data(
    collection: dataset.lob.LobReportCollector,
    key: str,
    level: int,
) -> list[int] | list[float]:
    data = []

    for _, report in collection.reports.items():
        try:
            data.append(report.integrity["nans"]["levels"][level][key])
        except KeyError:
            continue

    return data


def _collect_side_data(
    collection: dataset.lob.LobReportCollector,
    key: str,
    level: int,
    side: config.LobSide,
) -> list[int] | list[float]:
    data = []

    for _, report in collection.reports.items():
        try:
            data.append(report.integrity["nans"]["sides"][(level, side)][key])
        except KeyError:
            continue

    return data


def _collect_column_data(
    collection: dataset.lob.LobReportCollector,
    key: str,
    level: int,
    side: config.LobSide,
    column: config.LobColumn,
) -> list[int] | list[float]:
    col = config.LOB.get_column(
        level=level,
        side=side,
        column_type=column,
    )

    # The column integrity data only has the non-percentage / percentage
    # equivalent, hence the mapping from the requested key.
    temp_key = "n_nan_perc" if _is_perc(key) else "n_nan"

    data = []

    for _, report in collection.reports.items():
        try:
            data.append(report.integrity["nans"]["columns"][col][temp_key])
        except KeyError:
            continue

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _nans_hist_settings(
    key: str,
) -> tuple[range | str, str]:
    if _is_perc(key):
        return range(0, 101, 1), "Percent of NaNs"

    return "fd", "Number of NaNs"


def _plot_nans_hist(
    ax: plt.Axes,  # type: ignore (this is 'illegal' since plt.Axes is intended to be private)
    data: list[int] | list[float],
    key: str,
    title: str,
):
    bins, xlabel = _nans_hist_settings(key)

    ax.hist(
        data,
        bins=bins,
        align="left",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of dates")
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if _is_perc(key):
        ax.set_xticks(range(0, 105, 5))
        ax.set_xticks(range(0, 101, 1), minor=True)


def _plot_nans_hist_grid(
    data: dict[str, list[int] | list[float]],
    key: str,
    title: str,
    *,
    ncols: int = 2,
):
    if not data:
        return

    config.default_plt()

    nplots = len(data)
    ncols = min(ncols, nplots)
    nrows = (nplots + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 4 * nrows),
        squeeze=False,
    )

    axes = axes.flatten()

    for ax, (subplot_title, values) in zip(axes, data.items()):
        _plot_nans_hist(
            ax=ax,
            data=values,
            key=key,
            title=subplot_title,
        )

    # Hide unused axes.
    for ax in axes[len(data) :]:
        ax.set_visible(False)

    fig.suptitle(title)

    fig.tight_layout()
    plt.show()


def nans_hist_lob(
    collection: dataset.lob.LobReportCollector,
    key: str,
):
    data = _collect_lob_data(collection, key)

    title = (
        f"Distribution of the {N_NAN_TO_TITLE[key]} "
        f"in {collection.ticker.upper()} "
        f"{collection.role.upper()} LOB"
    )

    _plot_nans_hist_grid(
        {"LOB": data},
        key,
        title,
        ncols=1,
    )


def nans_hist_levels(
    collection: dataset.lob.LobReportCollector,
    key: str,
    *,
    ncols: int = 2,
):
    data = {}

    for level in config.LOB.levels:
        values = _collect_level_data(
            collection,
            key,
            level,
        )

        if values:
            data[f"Level {level}"] = values

    if not data:
        return

    title = (
        f"Distribution of the {N_NAN_TO_TITLE[key]} "
        f"in {collection.ticker.upper()} "
        f"{collection.role.upper()} LOB"
    )

    _plot_nans_hist_grid(
        data,
        key,
        title,
        ncols=ncols,
    )


def nans_hist_sides(
    collection: dataset.lob.LobReportCollector,
    key: str,
    level: int,
    *,
    ncols: int = 2,
):
    data = {}

    for side in get_args(config.LobSide):
        values = _collect_side_data(
            collection,
            key,
            level,
            side,
        )

        if values:
            data[f"Side {side.capitalize()}"] = values

    if not data:
        return

    title = (
        f"Distribution of the {N_NAN_TO_TITLE[key]} "
        f"in {collection.ticker.upper()} "
        f"{collection.role.upper()} LOB"
        f"\nLevel {level!r}"
    )

    _plot_nans_hist_grid(
        data,
        key,
        title,
        ncols=ncols,
    )


def nans_hist_side_columns(
    collection: dataset.lob.LobReportCollector,
    key: str,
    level: int,
    *,
    ncols: int = 2,
):
    data = {}

    for side in get_args(config.LobSide):
        for column in get_args(config.LobColumn):
            values = _collect_column_data(
                collection,
                key,
                level,
                side,
                column,
            )

            if values:
                data[f"Side {side.capitalize()} - " f"Column {column.capitalize()}"] = values

    if not data:
        return

    title = (
        f"Distribution of the {N_NAN_TO_TITLE[key]} "
        f"in {collection.ticker.upper()} "
        f"{collection.role.upper()} LOB"
        f"\nLevel {level!r}"
    )

    _plot_nans_hist_grid(
        data,
        key,
        title,
        ncols=ncols,
    )


def nans_hist_per_role(
    collection: dataset.lob.LobReportCollector,
    key: str,
):
    # Whole LOB
    nans_hist_lob(
        collection,
        key,
    )

    # All levels in one figure
    nans_hist_levels(
        collection,
        key,
    )

    # All sides in one figure per level
    for level in config.LOB.levels:
        nans_hist_sides(
            collection,
            key,
            level,
        )

    # All side x column combinations in one figure per level
    for level in config.LOB.levels:
        nans_hist_side_columns(
            collection,
            key,
            level,
        )


def nans_hist_per_ticker(
    collections: dict[config.AssetRole, dataset.lob.LobReportCollector],
    key: str,
):
    if key not in N_NAN_TO_TITLE:
        raise ValueError(f"Unknown {key=}")

    for collection in collections.values():
        nans_hist_per_role(
            collection,
            key,
        )
