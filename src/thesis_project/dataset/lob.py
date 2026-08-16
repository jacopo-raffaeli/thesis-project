import datetime
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from thesis_project import config, utils


@dataclass
class LobReport:
    date: datetime.date
    asset: config.AssetConfig
    modified: list[str] = field(default_factory=list[str])
    warning: list[str] = field(default_factory=list[str])
    critical: list[str] = field(default_factory=list[str])

    def print_modified(self):
        print("List of solved issues:")
        for v in self.modified:
            print(f"- {v}")

    def print_warning(self):
        print("List of warning issues:")
        for v in self.critical:
            print(f"- {v}")

    def print_critical(self):
        print("List of critical issues:")
        for v in self.critical:
            print(f"- {v}")

    def print_report(self):
        print(f"{self.date}: LOB {self.asset.symbol}")
        self.print_modified()
        self.print_warning()
        self.print_critical()


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
) -> tuple[pd.DataFrame, LobReport]:
    # Set the expected timezone
    assert isinstance(lob.index, pd.DatetimeIndex)

    if lob.index.tz is None:
        lob.index = lob.index.tz_localize(asset.market.tz)
        report.modified.append(f"LOB index: localized  timezone to {asset.market.tz.key}")

    elif str(lob.index.tz) != asset.market.tz.key:
        lob.index = lob.index.tz_convert(asset.market.tz)
        report.modified.append(f"LOB index: converted timezone to {asset.market.tz.key}")

    return lob, report


def _normalize_index(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> tuple[pd.DataFrame, LobReport]:
    # Sort index
    if not lob.index.is_monotonic_increasing:
        lob = lob.sort_index()
        report.modified.append("LOB index: sorted ascending")

    # Remove samples outside the market time
    assert isinstance(lob.index, pd.DatetimeIndex)
    mask = lob.index.indexer_between_time(asset.market.opening_time, asset.market.closing_time)
    n_outside = len(lob) - len(mask)
    if n_outside > 0:
        lob = lob.iloc[mask]
        report.modified.append(
            f"LOB: removed {n_outside} samples outside {asset.market.opening_time} - {asset.market.closing_time}"
        )

    # Remove index duplicates
    if lob.index.has_duplicates:
        mask = lob.index.duplicated()
        lob = lob[~mask]
        report.modified.append(f"LOB index: removed {mask.sum()} duplicates")

    # Remove index NaNs
    if lob.index.hasnans:
        mask = lob.index.isna().to_numpy()
        lob = lob[~mask]
        report.modified.append(f"LOB index: removed {mask.sum()} NaNs")

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
        report.modified.append("LOB index: adjusted to the expected time index ")

    # Rename the LOB index
    lob = lob.rename_axis(metadata.index_name)

    return lob, report


def _normalize_columns(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> tuple[pd.DataFrame, LobReport]:
    # Check that all the relevant columns are present
    expected = set(metadata.columns)
    present = set(lob.columns)

    missing = expected.difference(present)
    if missing:
        report.modified.append(
            f"LOB columns: {len(missing)} missing columns attached ({','.join(sorted(missing))})"
        )

    extra = present.difference(expected)
    if extra:
        report.modified.append(
            f"LOB columns: {len(extra)} extra columns dropped ({','.join(sorted(extra))})"
        )

    if missing or extra:
        lob = lob.reindex(columns=metadata.columns)

    return lob, report


def normalize(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> tuple[pd.DataFrame, LobReport]:
    """"""
    lob, report = _normalize_timezone(lob, report, metadata, asset)
    lob, report = _normalize_index(lob, report, metadata, asset)
    lob, report = _normalize_columns(lob, report, metadata, asset)

    return lob, report


def consistency(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> tuple[pd.DataFrame, LobReport]:
    """"""
    ...


def preprocess(path: Path, asset: config.AssetConfig, metadata: config.LobMetadata):
    """
    LOB preprocessing pipeline:
    * Load the parquet file to a DataFrame
    * Normalize to the expected LOB structure
    * ...

    ## Args:
    * path: path like "/data/raw/ticker/asset/yyyy/mm/asset_lob_freq_1s_yyyy_mm_dd.parquet"
    * asset: object of the class AssetConfig
    * metadata: object of the class LobMetadata

    ## Return:
    * ...
    """
    date = utils.io.filename_to_date(path.name)
    report = LobReport(date, asset)

    lob = load(path)
    lob, report = normalize(lob, report, metadata, asset)
    ...

    return lob, report
