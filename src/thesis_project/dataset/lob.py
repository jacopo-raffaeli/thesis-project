import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from thesis_project import config, utils

# TODO: Add reporting where needed
# TODO: Split in unit functions the
# TODO: Add some integrity check about NaN gaps in the lob


LobRecordType = Literal[
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


@dataclass(frozen=True)
class LobRecord:
    id: LobRecordType
    description: str
    timestamps: pd.DatetimeIndex | None = None
    columns: tuple[str, ...] | None = None


@dataclass
class LobReport:
    date: datetime.date
    asset: config.AssetConfig
    normalization: list[LobRecord] = field(default_factory=list)
    consistency: list[LobRecord] = field(default_factory=list)
    integrity: list[LobRecord] = field(default_factory=list)


def load(path: Path) -> pd.DataFrame:
    """
    Load a LOB parquet file and perform preliminary checks

    ## Args:
    * path: path like "/data/raw/ticker/asset/yyyy/mm/asset_lob_freq_1s_yyyy_mm_dd.parquet"

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

    if not isinstance(lob.index, pd.DatetimeIndex):
        raise TypeError("The LOB index is not a DatetimeIndex")

    return lob


def _normalize_timezone(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
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
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
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
            LobRecord(id="INDEX_DROP_NANS", description=f"Lob index: dropped {len(mask)} NaNs")
        )

    # Remove index duplicates
    if lob.index.has_duplicates:
        mask = lob.index.duplicated(keep="first")
        lob = lob[~mask]
        report.normalization.append(
            LobRecord(
                id="INDEX_DROP_DUPS", description=f"Lob index: dropped {len(mask)} duplicates"
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
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> pd.DataFrame:
    expected = set(metadata.columns())
    present = set(lob.columns)
    missing = expected.difference(present)
    extra = present.difference(expected)

    if missing:
        report.normalization.append(
            LobRecord(
                id="COLUMNS_MISSING_ADD",
                description=f"LOB columns: add missing columns ({','.join(sorted(missing))})",
            )
        )

    if extra:
        report.normalization.append(
            LobRecord(
                id="COLUMNS_EXTRA_DROP",
                description=f"LOB columns: dropped extra columns ({','.join(sorted(extra))})",
            )
        )

    if missing or extra:
        lob = lob.reindex(columns=metadata.columns())

    return lob


def normalize(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
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
    * metadata: object of the class LobMetadata
    * asset: object of the class AssetConfig

    ## Return:
    * lob: normalized LOB DataFrame
    """
    lob = _normalize_timezone(lob, report, metadata, asset)
    lob = _normalize_index(lob, report, metadata, asset)
    lob = _normalize_columns(lob, report, metadata, asset)

    return lob


def _consistency_price_sign(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = metadata.columns(column_types="price")
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
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = metadata.columns(column_types="price")
    for column, s in lob[columns].items():
        s = s.dropna()
        quotients = s / asset.tick_size_perc
        mask = ~np.isclose(quotients, np.round(quotients))
        ts = s.index[mask]
        if len(ts):
            report.consistency.append(
                LobRecord(
                    id="PRICE_NON_MULTIPLE",
                    description=f"LOB {str(column)}: found {len(ts)} non-multiple of {asset.tick_size_perc} prices",
                    timestamps=pd.DatetimeIndex(ts),
                    columns=(str(column),),
                )
            )


def _consistency_price_bid_order(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = [metadata.bid_price_column.format(level=level) for level in metadata.levels]
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
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = [metadata.ask_price_column.format(level=level) for level in metadata.levels]
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
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    col_bid = metadata.bid_price_column.format(level=1)
    col_ask = metadata.ask_price_column.format(level=1)
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
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    _consistency_price_sign(lob, report, metadata, asset)
    _consistency_price_unit(lob, report, metadata, asset)
    _consistency_price_bid_order(lob, report, metadata, asset)
    _consistency_price_ask_order(lob, report, metadata, asset)
    _consistency_price_bid_ask_order(lob, report, metadata, asset)


def _consistency_size_sign(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = metadata.columns(column_types="size")
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
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = metadata.columns(column_types="size")
    for column, s in lob[columns].items():
        s = s.dropna()
        mask = ~np.isclose(s, np.round(s))
        ts = s.index[mask]
        if len(ts):
            report.consistency.append(
                LobRecord(
                    id="SIZE_NON_MULTIPLE",
                    description=f"LOB {str(column)}: found {len(ts)} non-multiple of {1} lot sizes",
                    timestamps=pd.DatetimeIndex(ts),
                    columns=(str(column),),
                )
            )


def _consistency_size(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    _consistency_size_sign(lob, report, metadata, asset)
    _consistency_size_unit(lob, report, metadata, asset)


def _consistency_price_size(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    for level in metadata.levels:
        for side in metadata.sides:
            price_column = metadata.columns(levels=level, sides=side, column_types="price")[0]
            size_column = metadata.columns(levels=level, sides=side, column_types="size")[0]
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
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
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
    * metadata: object of the class LobMetadata
    * asset: object of the class AssetConfig
    """
    _consistency_price(lob, report, metadata, asset)
    _consistency_size(lob, report, metadata, asset)
    _consistency_price_size(lob, report, metadata, asset)


def _integrity_nans(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    info = {}
    n_elem = lob.size
    n_rows, n_cols = lob.shape
    na = lob.isna()

    # Shape
    info.update(
        {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_elem": n_elem,
        }
    )

    # LOB integrity
    info.update(
        {
            "lob_n_nan": na.sum().sum(),
            "lob_n_nan_perc": na.mean().mean() * 100,
        }
    )

    # Level integrity
    for level in metadata.levels:
        columns = metadata.columns(levels=level)
        info.update(
            {
                f"level_{level}_n_nan": na[columns].sum().sum(),
                f"level_{level}_n_nan_perc": na[columns].mean().mean() * 100,
            }
        )

    # Side integrity
    for level in metadata.levels:
        for side in metadata.sides:
            columns = metadata.columns(levels=level, sides=side)
            info.update(
                {
                    f"level_{level}_side_{side}_n_nan": na[columns].sum().sum(),
                    f"level_{level}_side_{side}_n_nan_perc": na[columns].mean().mean() * 100,
                }
            )

    # TODO: Decide whether to keep or not the info below
    # Column integrity
    for column in metadata.columns():
        info.update(
            {
                f"{column.lower().replace('-', '_')}_n_nan": na[column].sum(),
                f"{column.lower().replace('-', '_')}_n_nan_perc": na[column].mean() * 100,
            }
        )

    # TODO: Decide whether to keep or not the info below
    # Rows integrity
    info.update(
        {
            "n_rows_all_nan": na.all(axis=1).sum(),
            "n_rows_all_nan_perc": na.all(axis=1).mean() * 100,
            "n_rows_any_nan": na.any(axis=1).sum(),
            "n_rows_any_nan_perc": na.any(axis=1).mean() * 100,
        }
    )

    # TODO: Decide whether to keep or not the info below
    # Cols integrity
    info.update(
        {
            "n_cols_all_nan": na.all(axis=0).sum(),
            "n_cols_all_nan_perc": na.all(axis=0).mean() * 100,
            "n_cols_any_nan": na.any(axis=0).sum(),
            "n_cols_any_nan_perc": na.any(axis=0).mean() * 100,
        }
    )

    # TODO: Integrate info to report
    ...


def _integrity_valid_ts(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    info = {}
    filtered = lob.dropna()

    # LOB-wise valid index
    info.update(
        {
            "lob_min_valid_idx": filtered.index[0] if not filtered.empty else None,
            "lob_max_valid_idx": filtered.index[-1] if not filtered.empty else None,
        }
    )

    # Level-wise valid index
    for level in metadata.levels:
        columns = metadata.columns(levels=level)
        filtered = lob[columns].dropna()
        info.update(
            {
                f"level_{level}_min_valid_idx": filtered.index[0] if not filtered.empty else None,
                f"level_{level}_max_valid_idx": filtered.index[-1] if not filtered.empty else None,
            }
        )

    # Side-wise valid index
    for level in metadata.levels:
        for side in metadata.sides:
            columns = metadata.columns(levels=level, sides=side)
            filtered = lob[columns].dropna()
            info.update(
                {
                    f"level_{level}_side_{side}_min_valid_idx": filtered.index[0]
                    if not filtered.empty
                    else None,
                    f"level_{level}_side_{side}_max_valid_idx": filtered.index[-1]
                    if not filtered.empty
                    else None,
                }
            )

    # TODO: Integrate info to report
    ...


def integrity(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    """
    LOB integrity pipeline:
    * NaNs checks:
        * LOB-wise NaNs
        * Level-wise NaNs
        * Side-wise NaNs
        * Column-wise NaNs
        * Rows-wise NaNs
        * Columns-wise NaNs
    * Valid timestamps checks:
        * LOB-wise min/max valid timestamps
        * Level-wise min/max valid timestamps
        * Side-wise min/max valid timestamps

    ## Args:
    * lob: DataFrame containing the LOB
    * report: object of the class LobReport
    * metadata: object of the class LobMetadata
    * asset: object of the class AssetConfig
    """
    _integrity_nans(lob, report, metadata, asset)
    _integrity_valid_ts(lob, report, metadata, asset)
    ...


def preprocess(path: Path, asset: config.AssetConfig, metadata: config.LobMetadata):
    """
    LOB preprocessing pipeline:
    * Load the parquet file to a DataFrame
    * Normalize to the expected LOB structure
    * Consistency checks
    * Integrity checks

    ## Args:
    * path: path like "/data/raw/ticker/asset/yyyy/mm/asset_lob_freq_1s_yyyy_mm_dd.parquet"
    * asset: object of the class AssetConfig
    * metadata: object of the class LobMetadata

    ## Return:
    * lob: normalized LOB DataFrame
    * report: object containing a detailed report of the preprocessing
    """
    date = utils.io.filename_to_date(path.name)
    report = LobReport(date, asset)

    lob = load(path)
    lob = normalize(lob, report, metadata, asset)
    consistency(lob, report, metadata, asset)
    integrity(lob, report, metadata, asset)

    return lob, report
