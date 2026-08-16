import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_project import config, utils

# TODO: Implement in LobMetadata regex/formattable for the columns
# TODO: Implement a more useful LobReport daataclass
# TODO: Add reporting where needed
# TODO: Write missing function description
# TODO: Split in unit functions the checks


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

    elif str(lob.index.tz) != asset.market.tz.key:
        lob.index = lob.index.tz_convert(asset.market.tz)

    return lob, report


def _normalize_index(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> tuple[pd.DataFrame, LobReport]:
    # Sort index
    if not lob.index.is_monotonic_increasing:
        lob = lob.sort_index()

    # Remove samples outside the market time
    assert isinstance(lob.index, pd.DatetimeIndex)
    mask = lob.index.indexer_between_time(asset.market.opening_time, asset.market.closing_time)
    n_outside = len(lob) - len(mask)
    if n_outside > 0:
        lob = lob.iloc[mask]

    # Remove index duplicates
    if lob.index.has_duplicates:
        mask = lob.index.duplicated()
        lob = lob[~mask]

    # Remove index NaNs
    if lob.index.hasnans:
        mask = lob.index.isna().to_numpy()
        lob = lob[~mask]

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
    extra = present.difference(expected)

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


def _consistency_price(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> tuple[pd.DataFrame, LobReport]:
    # Check that prices are all positive
    columns = [c for c in lob.columns if "Price" in c]
    for k, s in lob[columns].items():
        s = s.dropna()
        mask = s <= 0
        if mask.any():
            # TODO: column k has mask.sum() non-positive prices
            ...

    # Check that prices are all multiples of 1 tick
    columns = [c for c in lob.columns if "Price" in c]
    for k, s in lob[columns].items():
        s = s.dropna()
        quotients = s / asset.tick_size_perc
        mask = ~np.isclose(quotients, np.round(quotients))
        if mask.any():
            # TODO: column k has mask.sum() non-multiple prices
            ...

    # Check that bid prices are ordered
    pattern = re.compile(r"^L(\d+)-BidPrice$")
    columns = [(int(m.group(1)), c) for c in lob.columns if (m := pattern.match(c))]
    columns.sort()

    levels = [level for level, _ in columns]
    columns = [column for _, column in columns]

    if not levels == metadata.levels:
        raise ValueError(
            f"Unexpected LOB bid prices level list: {','.join([str(level) for level in levels])}"
        )

    df = lob[columns].dropna()
    for c1, c2 in zip(columns[:-1], columns[1:]):
        s1 = df[c1]
        s2 = df[c2]
        mask = s1 <= s2
        if mask.any():
            # TODO: levels x,y has mask.sum() unordered prices
            ...

    # Check that ask prices are ordered
    pattern = re.compile(r"^L(\d+)-AskPrice$")
    columns = [(int(m.group(1)), c) for c in lob.columns if (m := pattern.match(c))]
    columns.sort()

    levels = [level for level, _ in columns]
    columns = [column for _, column in columns]

    if not levels == metadata.levels:
        raise ValueError(
            f"Unexpected LOB ask prices level list: {','.join([str(level) for level in levels])}"
        )

    df = lob[columns].dropna()
    for c1, c2 in zip(columns[:-1], columns[1:]):
        s1 = df[c1]
        s2 = df[c2]
        mask = s1 >= s2
        if mask.any():
            # TODO: levels x,y has mask.sum() unordered prices
            ...

    # Check that best bid < best ask
    s_bid = lob["L1-BidPrice"]
    s_ask = lob["L1-AskPrice"]

    mask = s_bid.isna() or s_ask.isna()
    s_bid = s_bid[mask]
    s_ask = s_ask[mask]

    mask = s_bid >= s_ask
    if mask.any():
        # TODO: Best bid and ask unordred for amsk.sum() seconds
        ...

    return lob, report


def _consistency_size(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> tuple[pd.DataFrame, LobReport]:
    # Check that sizes are all positive
    columns = [c for c in lob.columns if "Size" in c]
    for k, s in lob[columns].items():
        s = s.dropna()
        mask = s <= 0
        if mask.any():
            ...
            # TODO: column k has mask.sum() non-positive sizes

    # Check that sizes are all multiples of 1
    columns = [c for c in lob.columns if "Size" in c]
    for k, s in lob[columns].items():
        s = s.dropna()
        mask = ~np.isclose(s, np.round(s))
        if mask.any():
            ...
            # TODO: column k has mask.sum() non-multiple sizes
    ...

    return lob, report


def consistency(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> tuple[pd.DataFrame, LobReport]:
    """"""
    _consistency_price(lob, report, metadata, asset)
    _consistency_size(lob, report, metadata, asset)
    ...

    return lob, report


def integrity():
    """"""
    ...


def preprocess(path: Path, asset: config.AssetConfig, metadata: config.LobMetadata):
    """
    LOB preprocessing pipeline:
    * Load the parquet file to a DataFrame
    * Normalize to the expected LOB structure
    * Consistency check on the values

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
    lob, report = consistency(lob, report, metadata, asset)
    ...

    return lob, report
