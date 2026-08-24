import datetime
import logging
import pickle as pkl
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd

from thesis_project import config, utils

logger = logging.getLogger(__name__)


# TODO: Add utilities to LobReportCollector
# TODO: Add utilities to generate dates to exclude (maybe better in utils than here)
# TODO: Add number of levels to LobReportCollector or preprocess()

LobRecordTypeNormalization = Literal[
    # LOB index timezone
    "TIMEZONE_LOCALIZED",
    "TIMEZONE_CONVERTED",
    # LOB index
    "INDEX_SORTED",
    "INDEX_WINDOW_CUT",
    "INDEX_DROP_NANS",
    "INDEX_DROP_DUPS",
    "INDEX_GRID_ADJUSTED",
    # LOB columns
    "COLUMNS_EXTRA_DROP",
    "COLUMNS_MISSING_ADD",
]


LobRecordTypeConsistency = Literal[
    # Price consistency
    "PRICE_NON_POSITIVE",
    "PRICE_NON_MULTIPLE",
    "PRICE_BID_NON_ORDERED",
    "PRICE_ASK_NON_ORDERED",
    "PRICE_BID_ASK_NON_ORDERED",
    # Size consistency
    "SIZE_NON_POSITIVE",
    "SIZE_NON_MULTIPLE",
    # Price-Size consistency
    "PRICE_SIZE_PARTIAL",
]


LobRecordType = Literal[
    # NORMALIZATION
    LobRecordTypeNormalization,
    # CONSISTENCY
    LobRecordTypeConsistency,
]


@dataclass(frozen=True)
class LobRecord:
    id: LobRecordType
    description: str
    timestamps: pd.DatetimeIndex | None = None
    columns: tuple[str, ...] | None = None

    def __str__(self) -> str:
        return self.description

    @property
    def has_timestamps(self) -> bool:
        return self.timestamps is not None and len(self.timestamps) > 0

    @property
    def n_timestamps(self) -> int:
        return 0 if self.timestamps is None else len(self.timestamps)

    @property
    def first_timestamp(self) -> pd.Timestamp | None:
        if not self.has_timestamps:
            return None
        assert isinstance(self.timestamps, pd.DatetimeIndex)
        return self.timestamps[0]

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        if not self.has_timestamps:
            return None
        assert isinstance(self.timestamps, pd.DatetimeIndex)
        return self.timestamps[-1]


@dataclass
class LobReport:
    date: datetime.date
    asset: config.AssetConfig
    normalization: list[LobRecord] = field(default_factory=list)
    consistency: list[LobRecord] = field(default_factory=list)
    integrity: dict[str, Any] = field(default_factory=dict)

    @property
    def has_normalization_records(self) -> bool:
        return bool(self.normalization)

    @property
    def has_consistency_records(self) -> bool:
        return bool(self.consistency)

    @property
    def has_records(self) -> bool:
        return self.has_normalization_records or self.has_consistency_records

    @property
    def records(self) -> list[LobRecord]:
        return self.normalization + self.consistency

    def records_by_id(self, id: LobRecordType) -> list[LobRecord]:
        return [record for record in self.records if record.id == id]

    def normalization_summary(self):
        print("Normalization summary:")
        for record in self.normalization:
            print(record)

    def consistency_summary(self):
        print("Consistency summary:")
        for record in self.consistency:
            print(record)

    def summary(self):
        print(f"LOB {self.date} {self.asset.role} summary:")
        print()
        self.normalization_summary()
        print()
        self.consistency_summary()


@dataclass
class LobReportCollector:
    ticker: config.FutTicker
    role: config.AssetRole
    reports: dict[datetime.date, LobReport] = field(default_factory=dict)

    def add(self, report: LobReport):
        if report.asset.role != self.role:
            raise ValueError(
                f"Cannot add report for role {report.asset.role!r} " f"to {self.role!r} collector."
            )

        if report.date in self.reports:
            raise ValueError(
                f"Duplicate report for {self.ticker=} {report.asset.role=} {report.date=}"
            )

        self.reports[report.date] = report

    def summary(self):
        summary: dict[LobRecordType, int] = {id: 0 for id in get_args(LobRecordType)}
        for report in self.reports.values():
            for record in report.records:
                summary[record.id] += 1

        print(f"Report summary for {self.ticker.upper()} {self.role.upper()} LOBs:")
        print(f"- Number of dates: {len(self.reports)}")
        print("- Issues count:")
        for k, v in summary.items():
            print(f"  - {k}: {v}")

    def dates_by_id(self, id: LobRecordType):
        foo = {}
        for report in self.reports.values():
            for record in report.records:
                if record == id:
                    foo[report.date]


def load(path: Path) -> pd.DataFrame:
    """
    Load a LOB parquet file and perform preliminary checks

    ## Args:
    * path: "/data/raw/{ticker}/{asset}/yyyy/mm/{asset}_lob_freq_1s_yyyy_mm_dd.parquet"

    ## Return:
    * lob: DataFrame containing the LOB data
    """
    # Load LOB
    lob = pd.read_parquet(path)

    # Preliminary checks
    if not isinstance(lob, pd.DataFrame):
        raise TypeError("The LOB is not a DataFrame")

    if lob.empty:
        raise ValueError("The LOB is empty")

    logger.debug("LOB loaded: name=%s, rows=%d, columns=%d", path.name, len(lob), len(lob.columns))

    return lob


def set_index(lob: pd.DataFrame, metadata: config.LobConfig) -> pd.DataFrame:
    # Case 1:
    # We already have a DatetimeIndex set
    # We set the default name whichever is the actual one
    if isinstance(lob.index, pd.DatetimeIndex):
        lob = lob.rename_axis(metadata.index_name)
        return lob

    CANDIDATES = ("timestamp", "Date-Time")
    present = [c for c in CANDIDATES if c in lob.columns]

    if not present:
        raise ValueError(
            "The LOB has neither a DatetimeIndex nor a timestamp column. "
            f"Expected one of: {', '.join(CANDIDATES)}"
        )

    if len(present) > 1:
        raise ValueError(f"The LOB contains multiple timestamp candidates: {present}")

    column = present[0]
    try:
        index = pd.DatetimeIndex(pd.to_datetime(lob[column]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not convert LOB timestamp column {column!r} to datetime") from exc

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"Timestamp column {column!r} did not produce a DatetimeIndex")

    print("SET_INDEX FILE:", __file__)
    print("COLUMN:", column)
    print("BEFORE DROP:", lob.columns.tolist())

    lob = lob.drop(columns=column)

    print("AFTER DROP:", lob.columns.tolist())

    lob.index = index
    return lob.rename_axis(metadata.index_name)


def _normalize_timezone(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
) -> pd.DataFrame:
    assert isinstance(lob.index, pd.DatetimeIndex)

    if lob.index.tz is None:
        report.normalization.append(
            LobRecord(
                id="TIMEZONE_LOCALIZED",
                description=f"LOB index timezone: localized from None to {asset.market.tz.key}",
            )
        )
        lob.index = lob.index.tz_localize(asset.market.tz)

    elif str(lob.index.tz) != asset.market.tz.key:
        report.normalization.append(
            LobRecord(
                id="TIMEZONE_CONVERTED",
                description=f"LOB index timezone: converted from {str(lob.index.tz)} to {asset.market.tz.key}",
            )
        )
        lob.index = lob.index.tz_convert(asset.market.tz)

    return lob


def _normalize_index(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
) -> pd.DataFrame:
    # Sort index
    if not lob.index.is_monotonic_increasing:
        lob = lob.sort_index(ascending=True)
        report.normalization.append(
            LobRecord(id="INDEX_SORTED", description="LOB index: sorted in ascending order")
        )

    # Remove samples outside the market time
    assert isinstance(lob.index, pd.DatetimeIndex)
    mask = lob.index.indexer_between_time(asset.market.opening_time, asset.market.closing_time)
    n_outside = len(lob) - len(mask)
    if n_outside > 0:
        lob = lob.iloc[mask]
        report.normalization.append(
            LobRecord(
                id="INDEX_WINDOW_CUT",
                description=f"Lob index: dropped {n_outside} timestamps outisde {asset.market.opening_time} - {asset.market.closing_time} window",
            )
        )

    # Remove index NaNs
    if lob.index.hasnans:
        mask = lob.index.isna().to_numpy()
        lob = lob[~mask]
        report.normalization.append(
            LobRecord(id="INDEX_DROP_NANS", description=f"Lob index: dropped {mask.sum()} NaNs")
        )

    # Remove index duplicates
    if lob.index.has_duplicates:
        mask = lob.index.duplicated(keep="first")
        lob = lob[~mask]
        report.normalization.append(
            LobRecord(
                id="INDEX_DROP_DUPS", description=f"Lob index: dropped {mask.sum()} duplicates"
            )
        )

    # Set the expected time index
    start = datetime.datetime.combine(
        report.date,
        asset.market.opening_time,
        asset.market.tz,
    )

    end = datetime.datetime.combine(
        report.date,
        asset.market.closing_time,
        asset.market.tz,
    )

    index = pd.date_range(
        start=start,
        end=end,
        freq=metadata.freq,
    )

    if not lob.index.equals(index):
        lob = lob.reindex(index)
        report.normalization.append(
            LobRecord(
                id="INDEX_GRID_ADJUSTED",
                description=f"Lob index: reindexed to {metadata.freq} grid in {asset.market.opening_time} - {asset.market.closing_time} window",
            )
        )

    # Rename the LOB index
    lob = lob.rename_axis(metadata.index_name)

    return lob


def _normalize_columns(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
) -> pd.DataFrame:
    expected = set(metadata.get_columns())
    present = set(lob.columns)
    missing = expected.difference(present)
    extra = present.difference(expected)

    if missing:
        report.normalization.append(
            LobRecord(
                id="COLUMNS_MISSING_ADD",
                description=f"LOB columns: add missing columns ({', '.join(sorted(missing))})",
            )
        )

    if extra:
        report.normalization.append(
            LobRecord(
                id="COLUMNS_EXTRA_DROP",
                description=f"LOB columns: dropped extra columns ({', '.join(sorted(extra))})",
            )
        )

    if missing or extra:
        lob = lob.reindex(columns=metadata.get_columns())

    return lob


def normalize(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
) -> pd.DataFrame:
    """
    LOB normalization pipeline:
    * Normalize LOB index timezone:
        * Set LOB index timezone if None
        * Convert LOB index timezone if not default
    * Normalize the LOB index :
        * Sort index
        * Remove garbage outside the time window
        * Remove duplicates in the index
        * Remove NaNs in the index
        * Adjust to the expected time index grid (fill with NaNs)
        * Rename time index
    * Normalize the LOB columns names:
        * Drop extra columns
        * Add missing columns (fill with NaNs)

    ## Args:
    * lob: DataFrame containing the LOB
    * report: object of the class LobReport
    * metadata: object of the class LobConfig
    * asset: object of the class AssetConfig

    ## Return:
    * lob: normalized LOB DataFrame
    """
    lob = _normalize_timezone(lob, report, metadata, asset)
    lob = _normalize_index(lob, report, metadata, asset)
    lob = _normalize_columns(lob, report, metadata, asset)

    if report.has_normalization_records:
        logger.debug(
            "- LOB normalization corrections: asset=%s, date=%s, records=%d",
            asset.symbol,
            report.date,
            len(report.normalization),
        )

    return lob


def _consistency_price_sign(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    columns = metadata.get_columns(column_types="price")
    for column, s in lob[columns].items():
        s = s.dropna()
        mask = s <= 0
        ts = s.index[mask]
        if len(ts):
            report.consistency.append(
                LobRecord(
                    id="PRICE_NON_POSITIVE",
                    description=f"LOB {str(column)}: found {len(ts)} non-positive prices",
                    timestamps=pd.DatetimeIndex(ts),
                    columns=(str(column),),
                )
            )


def _consistency_price_unit(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    columns = metadata.get_columns(column_types="price")
    for column, s in lob[columns].items():
        s = s.dropna()
        quotients = s / asset.price_tick_perc
        mask = ~np.isclose(quotients, np.round(quotients))
        ts = s.index[mask]
        if len(ts):
            report.consistency.append(
                LobRecord(
                    id="PRICE_NON_MULTIPLE",
                    description=f"LOB {str(column)}: found {len(ts)} prices non-multiple of {asset.price_tick_perc} price tick",
                    timestamps=pd.DatetimeIndex(ts),
                    columns=(str(column),),
                )
            )


def _consistency_price_bid_order(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    columns = metadata.get_columns(sides="bid", column_types="price")
    for c1, c2 in zip(columns[:-1], columns[1:]):
        s1 = lob[c1]
        s2 = lob[c2]
        valid = s1.notna() & s2.notna()
        s1 = s1[valid]
        s2 = s2[valid]
        mask = s1 <= s2
        ts = s1.index[mask]
        if len(ts):
            report.consistency.append(
                LobRecord(
                    id="PRICE_BID_NON_ORDERED",
                    description=f"LOB {c1} - {c2}: found {len(ts)} non-ordered bid prices",
                    timestamps=pd.DatetimeIndex(ts),
                    columns=(c1, c2),
                )
            )


def _consistency_price_ask_order(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    columns = metadata.get_columns(sides="ask", column_types="price")
    for c1, c2 in zip(columns[:-1], columns[1:]):
        s1 = lob[c1]
        s2 = lob[c2]
        valid = s1.notna() & s2.notna()
        s1 = s1[valid]
        s2 = s2[valid]
        mask = s1 >= s2
        ts = s1.index[mask]
        if len(ts):
            report.consistency.append(
                LobRecord(
                    id="PRICE_ASK_NON_ORDERED",
                    description=f"LOB {c1} - {c2}: found {len(ts)} non-ordered ask prices",
                    timestamps=pd.DatetimeIndex(ts),
                    columns=(c1, c2),
                )
            )


def _consistency_price_bid_ask_order(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    col_bid = metadata.get_column(level=1, side="bid", column_type="price")
    col_ask = metadata.get_column(level=1, side="ask", column_type="price")
    s_bid = lob[col_bid]
    s_ask = lob[col_ask]

    valid = s_bid.notna() & s_ask.notna()
    s_bid = s_bid[valid]
    s_ask = s_ask[valid]

    mask = s_bid >= s_ask
    ts = s_bid.index[mask]
    if len(ts):
        report.consistency.append(
            LobRecord(
                id="PRICE_BID_ASK_NON_ORDERED",
                description=f"LOB {col_bid} - {col_ask}: found {len(ts)} non-ordered best bid-ask prices",
                timestamps=pd.DatetimeIndex(ts),
                columns=(col_bid, col_ask),
            )
        )


def _consistency_price(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    _consistency_price_sign(lob, report, metadata, asset)
    _consistency_price_unit(lob, report, metadata, asset)
    _consistency_price_bid_order(lob, report, metadata, asset)
    _consistency_price_ask_order(lob, report, metadata, asset)
    _consistency_price_bid_ask_order(lob, report, metadata, asset)


def _consistency_size_sign(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    columns = metadata.get_columns(column_types="size")
    for column, s in lob[columns].items():
        s = s.dropna()
        mask = s <= 0
        ts = s.index[mask]
        if len(ts):
            report.consistency.append(
                LobRecord(
                    id="SIZE_NON_POSITIVE",
                    description=f"LOB {str(column)}: found {len(ts)} non-positive sizes",
                    timestamps=pd.DatetimeIndex(ts),
                    columns=(str(column),),
                )
            )


def _consistency_size_unit(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    columns = metadata.get_columns(column_types="size")
    for column, s in lob[columns].items():
        s = s.dropna()
        quotients = s / asset.size_tick
        mask = ~np.isclose(quotients, np.round(quotients))
        ts = s.index[mask]
        if len(ts):
            report.consistency.append(
                LobRecord(
                    id="SIZE_NON_MULTIPLE",
                    description=f"LOB {str(column)}: found {len(ts)} sizes non-multiple of {asset.size_tick} size tick",
                    timestamps=pd.DatetimeIndex(ts),
                    columns=(str(column),),
                )
            )


def _consistency_size(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    _consistency_size_sign(lob, report, metadata, asset)
    _consistency_size_unit(lob, report, metadata, asset)


def _consistency_price_size(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    for level in metadata.levels:
        for side in metadata.sides:
            price_column = metadata.get_column(level=level, side=side, column_type="price")
            size_column = metadata.get_column(level=level, side=side, column_type="size")
            mask = lob[price_column].isna() ^ lob[size_column].isna()
            ts = lob.index[mask]
            if len(ts):
                report.consistency.append(
                    LobRecord(
                        id="PRICE_SIZE_PARTIAL",
                        description=f"LOB {price_column} - {size_column}: found {len(ts)} partial price-size pairs",
                        timestamps=pd.DatetimeIndex(ts),
                        columns=(price_column, size_column),
                    )
                )


def consistency(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    """
    LOB consistency pipeline:
    * Price consistency checks:
        * Check for non-positive prices
        * Check for non multiple of 1 tick prices
        * Check for unordered bid prices
        * Check for unordered ask prices
        * Check for unordered best bid-ask prices
    * Volume consistency checks:
        * Check for non-positive volumes
        * Check for non multiple of 1 lot prices
    * Price-Volume consistency checks:
        * Check for partial price-size pairs

    ## Args:
    * lob: DataFrame containing the LOB
    * report: object of the class LobReport
    * metadata: object of the class LobConfig
    * asset: object of the class AssetConfig
    """
    _consistency_price(lob, report, metadata, asset)
    _consistency_size(lob, report, metadata, asset)
    _consistency_price_size(lob, report, metadata, asset)

    if report.has_consistency_records:
        logger.debug(
            "- LOB consistency issues: asset=%s, date=%s, records=%d",
            asset.symbol,
            report.date,
            len(report.consistency),
        )


def _integrity_shape(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    n_rows, n_cols = lob.shape

    # Shape
    info = {
        "rows": n_rows,
        "columns": n_cols,
        "size": lob.size,
    }
    report.integrity["shape"] = info


def _integrity_nans(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    na = lob.isna()
    info = {
        "lob": {},
        "levels": {},
        "sides": {},
        "columns": {},
    }

    # LOB integrity
    info["lob"] = {
        "n_nan": na.sum().sum(),
        "n_nan_perc": na.mean().mean() * 100,
        "n_rows_all_nan": na.all(axis=1).sum(),
        "n_rows_all_nan_perc": na.all(axis=1).mean() * 100,
        "n_rows_any_nan": na.any(axis=1).sum(),
        "n_rows_any_nan_perc": na.any(axis=1).mean() * 100,
    }

    levels = [1]

    # Per level integrity
    for level in levels:
        columns = metadata.get_columns(levels=level)
        info["levels"][level] = {
            "n_nan": na[columns].sum().sum(),
            "n_nan_perc": na[columns].mean().mean() * 100,
            "n_rows_all_nan": na[columns].all(axis=1).sum(),
            "n_rows_all_nan_perc": na[columns].all(axis=1).mean() * 100,
            "n_rows_any_nan": na[columns].any(axis=1).sum(),
            "n_rows_any_nan_perc": na[columns].any(axis=1).mean() * 100,
        }

    # Per side integrity
    for level in levels:
        for side in metadata.sides:
            columns = metadata.get_columns(levels=level, sides=side)
            info["sides"][level, side] = {
                "n_nan": na[columns].sum().sum(),
                "n_nan_perc": na[columns].mean().mean() * 100,
                "n_rows_all_nan": na[columns].all(axis=1).sum(),
                "n_rows_all_nan_perc": na[columns].all(axis=1).mean() * 100,
                "n_rows_any_nan": na[columns].any(axis=1).sum(),
                "n_rows_any_nan_perc": na[columns].any(axis=1).mean() * 100,
            }

    # Per column integrity
    for column in metadata.get_columns(levels=levels):
        info["columns"][column] = {
            "n_nan": na[column].sum(),
            "n_nan_perc": na[column].mean() * 100,
        }

    report.integrity["nans"] = info


def _integrity_timestamps(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    valid = lob.notna()
    info = {
        "lob": {},
        "levels": {},
        "sides": {},
        "columns": {},
    }

    # LOB valid index
    mask = valid.all(axis=1)
    info["lob"] = {
        "min_valid_idx": lob.index[mask][0] if mask.any() else None,
        "max_valid_idx": lob.index[mask][-1] if mask.any() else None,
    }

    levels = [1]

    # Per level valid index
    for level in levels:
        columns = metadata.get_columns(levels=level)
        mask = valid[columns].all(axis=1)
        info["levels"][level] = {
            "min_valid_idx": lob.index[mask][0] if mask.any() else None,
            "max_valid_idx": lob.index[mask][-1] if mask.any() else None,
        }

    # Per side valid index
    for level in levels:
        for side in metadata.sides:
            columns = metadata.get_columns(levels=level, sides=side)
            mask = valid[columns].all(axis=1)
            info["sides"][level, side] = {
                "min_valid_idx": lob.index[mask][0] if mask.any() else None,
                "max_valid_idx": lob.index[mask][-1] if mask.any() else None,
            }

    # Per column valid index
    for column in metadata.get_columns(levels=levels):
        mask = valid[column]
        info["columns"][column] = {
            "min_valid_index": lob.index[mask][0] if mask.any() else None,
            "max_valid_index": lob.index[mask][-1] if mask.any() else None,
        }

    report.integrity["timestamps"] = info


def _find_true_groups(s: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]] | None:
    """
    Find groups of consecutive True boolean in a series

    ## Args:
    * s: A Series of bool indexed by a DatetimeIndex

    ## Returns:
    * ts: A list of tuple each containing opening and closing timestamps of a group
    """

    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a DatetimeIndex")

    if not s.any():
        return []

    groups = s.ne(s.shift()).cumsum()
    ts = s[s].groupby(groups[s]).apply(lambda ts: (ts.index[0], ts.index[-1]))

    return ts.tolist()


def _integrity_gaps(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    na = lob.isna()
    info = {
        "lob": ...,
        "levels": {},
        "sides": {},
        "columns": {},
    }

    # LOB gaps
    info["lob"] = _find_true_groups(na.all(axis=1))

    levels = [1]

    # Per level gaps
    for level in levels:
        columns = metadata.get_columns(levels=level)
        info["levels"][level] = _find_true_groups(na[columns].all(axis=1))

    # Per side gaps
    for level in levels:
        for side in metadata.sides:
            columns = metadata.get_columns(levels=level, sides=side)
            info["sides"][level, side] = _find_true_groups(na[columns].all(axis=1))

    # Per column gaps
    for column in metadata.get_columns(levels=levels):
        info["columns"][column] = _find_true_groups(na[[column]].all(axis=1))

    report.integrity["gaps"] = info


def integrity(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobConfig, asset: config.AssetConfig
):
    """
    NOTE: The analysis is actually performed only at LOB level 1 for computational reasons

    LOB integrity pipeline:
    * Shape:
        * Rows
        * Columns
        * Size
    * NaNs checks:
        * LOB NaNs
        * Per level NaNs
        * Per side-wise NaNs
        * Per column-wise NaNs
    * Valid timestamps checks:
        * LOB min/max valid timestamps
        * Per level min/max valid timestamps
        * Per side min/max valid timestamps
        * Per column min/max valid timestamps
    * NaNs groups checks:
        * LOB NaNs groups
        * Per level NaNs groups
        * Per side NaNs groups
        * Per column NaNs groups

    ## Args:
    * lob: DataFrame containing the LOB
    * report: object of the class LobReport
    * metadata: object of the class LobConfig
    * asset: object of the class AssetConfig
    """
    _integrity_shape(lob, report, metadata, asset)
    _integrity_nans(lob, report, metadata, asset)
    _integrity_timestamps(lob, report, metadata, asset)
    _integrity_gaps(lob, report, metadata, asset)


def preprocess(path: Path, asset: config.AssetConfig, metadata: config.LobConfig):
    """
    LOB preprocessing pipeline:
    * Load the parquet file to a DataFrame
    * Normalize to the expected LOB structure
    * Consistency checks
    * Integrity checks

    ## Args:
    * path: path like "/data/raw/ticker/asset/yyyy/mm/asset_lob_freq_1s_yyyy_mm_dd.parquet"
    * asset: AssetConfig object
    * metadata: LobConfig object

    ## Return:
    * lob: normalized LOB DataFrame
    * report: object containing a detailed report of the preprocessing
    """
    date = utils.io.filename_to_date(path.name)
    report = LobReport(date, asset)

    lob = load(path)
    lob = set_index(lob, metadata)
    lob = normalize(lob, report, metadata, asset)
    consistency(lob, report, metadata, asset)
    integrity(lob, report, metadata, asset)

    logger.info(
        "- LOB preprocessed: asset=%s, date=%s, rows=%d, columns=%d",
        asset.symbol,
        report.date,
        len(lob),
        len(lob.columns),
    )

    return lob, report


def preprocess_all_lobs(*, ticker: config.FutTicker, metadata: config.LobConfig = config.LOB):
    """
    Preprocess all raw LOBs for a ticker and save them to processed/.

    A LobReportCollector containing the reports for all processed LOBs
    is also saved alongside the processed LOB data.

    ## Args:
    * ticker: FutTicker literal
    * metadata: LobConfig object
    """
    for role in get_args(config.AssetRole):
        # reports = LobReportCollector(ticker=ticker, role=role)
        asset = config.ASSET_BY_TICKER_ROLE[(ticker, role)]
        raw_paths = utils.io.list_lob_paths(root=config.DATA_RAW_DIR, ticker=ticker, role=role)

        report_root = config.DATA_PRO_DIR / ticker / role / "reports"

        if report_root.exists():
            shutil.rmtree(report_root)

        report_root.mkdir(parents=True)

        for raw_path in raw_paths:
            # Run preprocessing
            lob, report = preprocess(raw_path, asset, metadata)

            # Save lob to data/processed/... specular path
            pro_path = config.DATA_PRO_DIR / raw_path.relative_to(config.DATA_RAW_DIR)
            pro_path.parent.mkdir(parents=True, exist_ok=True)
            lob.astype("float32").to_parquet(pro_path)

            # Save report
            save_lob_report(report, ticker, role)

            del lob
            del report

        # Group report collection
        group_lob_reports(ticker, role)


def save_lob_reports(reports: LobReportCollector):
    path = config.DATA_PRO_DIR / reports.ticker / reports.role
    path.mkdir(parents=True, exist_ok=True)
    path = path / "lob_reports.pkl"

    with path.open("wb") as f:
        pkl.dump(reports, f)


def load_lob_reports(ticker: config.FutTicker, role: config.AssetRole) -> LobReportCollector:
    path = config.DATA_PRO_DIR / ticker / role / "lob_reports.pkl"

    with path.open("rb") as f:
        return pkl.load(f)


def save_lob_report(report: LobReport, ticker: config.FutTicker, role: config.AssetRole):
    path = config.DATA_PRO_DIR / ticker / role / "reports"
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"{report.date}.pkl"

    with path.open("wb") as f:
        pkl.dump(report, f)


def load_lob_report(
    ticker: config.FutTicker, role: config.AssetRole, date: datetime.date
) -> LobReport:
    path = config.DATA_PRO_DIR / ticker / role / "reports" / f"{date}.pkl"

    with path.open("rb") as f:
        return pkl.load(f)


def group_lob_reports(ticker: config.FutTicker, role: config.AssetRole):
    root = config.DATA_PRO_DIR / ticker / role / "reports"

    if not root.exists():
        raise FileNotFoundError(f"LOB report directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"LOB report path is not a directory: {root}")

    reports = LobReportCollector(ticker=ticker, role=role)
    for path in sorted(root.glob("*.pkl")):
        with path.open("rb") as f:
            report = pkl.load(f)

        if not isinstance(report, LobReport):
            raise TypeError(f"Expected LobReport in {path}, " f"got {type(report).__name__}")
        reports.add(report)

    save_lob_reports(reports)
    shutil.rmtree(root)
