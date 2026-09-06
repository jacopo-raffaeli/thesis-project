import datetime as dt
import logging
from dataclasses import dataclass
from typing import Literal, get_args

import pandas as pd

from thesis_project import config, utils
from thesis_project.dataset.lob_preprocessing import LobReportCollector, load_lob_reports

logger = logging.getLogger(__name__)

CriticalityCheck = Literal[
    "consistency",
    "nans",
    "window",
    "gaps",
]

DEFAULT_CHECKS: tuple[CriticalityCheck, ...] = (
    "consistency",
    "nans",
    "window",
    "gaps",
)


@dataclass(frozen=True)
class Settings:
    checks: tuple[CriticalityCheck, ...]
    min_time: dt.time
    max_time: dt.time
    nan_threshold: float
    sec_threshold: float
    relevant_columns: tuple[str, ...] | None


def _valid_timestamps(timestamps: pd.DatetimeIndex, settings: Settings) -> pd.DatetimeIndex:
    times = timestamps.time
    return timestamps[[(settings.min_time <= t <= settings.max_time) for t in times]]


def _consistency_rows(
    collection: LobReportCollector,
    settings: Settings,
    relevant_columns,  # TODO: Is not this already in settings?
):
    rows = []
    for date, report in collection.reports.items():
        for record in report.consistency:
            if (record.columns is None) or (not record.has_timestamps):
                continue

            # Check that all the columns in record are relevant
            columns = tuple(c for c in record.columns if c in relevant_columns)
            if set(record.columns) != set(columns):
                continue

            # Check that at least some timestamp in record are relevant
            assert isinstance(record.timestamps, pd.DatetimeIndex)
            timestamps = _valid_timestamps(record.timestamps, settings)
            if len(timestamps) == 0:
                continue

            rows.append(
                {
                    "Date": date,
                    "Role": collection.role,
                    "Check": record.id,
                }
            )
    return rows


def _nan_rows(collection: LobReportCollector, settings: Settings):
    rows = []
    threshold = settings.nan_threshold
    for date, report in collection.reports.items():
        value = report.integrity["nans"]["levels"][1]["n_rows_all_nan_perc"]
        if value >= threshold:
            rows.append(
                {
                    "Date": date,
                    "Role": collection.role,
                    "Check": "NANS_LEVEL_1",
                }
            )

    return rows


def _window_rows(collection: LobReportCollector, settings: Settings):
    rows = []
    for date, report in collection.reports.items():
        info = report.integrity["timestamps"]["levels"][1]
        opening, closing = info["min_valid_idx"], info["max_valid_idx"]

        if opening.time() <= settings.min_time and closing.time() >= settings.max_time:
            continue

        rows.append(
            {
                "Date": date,
                "Role": collection.role,
                "Check": "VALID_WINDOW",
            }
        )

    return rows


def _gap_rows(
    collection: LobReportCollector,
    settings: Settings,
):
    rows = []
    for date, report in collection.reports.items():
        relevant = []
        for opening, closing in report.integrity["gaps"]["levels"][1]:
            opening_boundary = opening.replace(
                hour=settings.min_time.hour,
                minute=settings.min_time.minute,
                second=settings.min_time.second,
                microsecond=0,
            )
            closing_boundary = closing.replace(
                hour=settings.max_time.hour,
                minute=settings.max_time.minute,
                second=settings.max_time.second,
                microsecond=0,
            )
            opening, closing = max(opening, opening_boundary), min(closing, closing_boundary)
            if closing < opening:
                continue
            length = (closing - opening).total_seconds() + 1
            if length > settings.sec_threshold:
                relevant.append((opening, closing, length))

        if relevant:
            rows.append(
                {
                    "Date": date,
                    "Role": collection.role,
                    "Check": "GAPS_LEVEL_1",
                }
            )

    return rows


def find_criticalities(
    ticker: config.FutTicker,
    settings: Settings,
) -> pd.DataFrame:
    """
    Return detailed criticalities found in the ticker's LOB reports.

    ## Args:
    * ticker: FutTicker
    * settings: Settings

    ## Return:
    * result: DataFrame containing a detailed report of critical dates
    """
    relevant_columns = set(
        settings.relevant_columns
        if settings.relevant_columns is not None
        else config.LOB.get_columns(levels=[1, 2, 3])
    )

    rows = []
    for role in get_args(config.AssetRole):
        logger.debug("Scanning %s %s LOB reports...", str(ticker).upper(), str(role).upper())
        collection = load_lob_reports(ticker, role)

        if "consistency" in settings.checks:
            rows.extend(_consistency_rows(collection, settings, relevant_columns))
        if "nans" in settings.checks:
            rows.extend(_nan_rows(collection, settings))
        if "window" in settings.checks:
            rows.extend(_window_rows(collection, settings))
        if "gaps" in settings.checks:
            rows.extend(_gap_rows(collection, settings))

        logger.debug("%s %s LOB reports scanned", str(ticker).upper(), str(role).upper())

    columns = [
        "Date",
        "Role",
        "Check",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(["Date", "Role", "Check"], ignore_index=True)

    logger.debug("Scan complete %d unique dates collected", result["Date"].nunique())

    return result


def get_critical_dates(
    ticker: config.FutTicker,
    settings: Settings | None = None,
) -> list[dt.date]:
    """
    Return unique dates affected by the selected checks.

    ## Args:
    * ticker: FutTicker
    * settings: Settings, if not provided return the default list

    ## Return:
    * dates: list of datetime.date
    """
    if settings is not None:
        return find_criticalities(ticker, settings)["Date"].drop_duplicates().to_list()

    else:
        return utils.io.load_criticalities(ticker)
