import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from thesis_project.config import AssetConfig, LobMetadata


@dataclass
class LobReport:
    date: datetime.date
    asset: AssetConfig
    modified: list[str] = field(default_factory=list[str])
    critical: list[str] = field(default_factory=list[str])

    def print_modified(self):
        print("List of solved issues:")
        for v in self.modified:
            print(f"- {v}")

    def print_critical(self):
        print("List of critical issues:")
        for v in self.critical:
            print(f"- {v}")

    def print_report(self):
        print(f"{self.date}: LOB {self.asset.symbol}")
        self.print_modified()
        self.print_critical()


def filename_to_date(filename: str) -> datetime.date:
    """
    Extract the date from a LOB parquet filename.

    ## Args:
    * filename: expected format {ctd/fut}_lob_freq_1s_yyyy_mm_dd.parquet

    ## Return
    * datetime.date object
    """
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})\.parquet$", filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")

    year, month, day = map(int, match.groups())
    return datetime.date(year, month, day)


def load_lob(path: Path) -> pd.DataFrame:
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


def normalize_lob(lob: pd.DataFrame, report: LobReport, metadata: LobMetadata, asset: AssetConfig):
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

    # Set the expected timezone
    assert isinstance(lob.index, pd.DatetimeIndex)

    if lob.index.tz is None:
        lob.index = lob.index.tz_localize(asset.market.tz)
        report.modified.append(f"LOB index: localized  timezone to {asset.market.tz.key}")

    elif str(lob.index.tz) != asset.market.tz.key:
        lob.index = lob.index.tz_convert(asset.market.tz)
        report.modified.append(f"LOB index: converted timezone to {asset.market.tz.key}")

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
    lob.rename(index={lob.index.name: metadata.index_name})

    # TODO: Check that all the relevant columns are present
    # TODO: Keep only the relevant columns

    return lob, report


def preprocess_lob(path: Path, asset: AssetConfig, metadata: LobMetadata):
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
    date = filename_to_date(path.name)
    report = LobReport(date, asset)

    lob = load_lob(path)
    lob, report = normalize_lob(lob, report, metadata, asset)

    return lob, report
