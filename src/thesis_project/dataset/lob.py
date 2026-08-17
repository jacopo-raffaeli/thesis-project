import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_project import config, utils

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
) -> pd.DataFrame:
    assert isinstance(lob.index, pd.DatetimeIndex)

    if lob.index.tz is None:
        lob.index = lob.index.tz_localize(asset.market.tz)
        # TODO: Report tz localization

    elif str(lob.index.tz) != asset.market.tz.key:
        lob.index = lob.index.tz_convert(asset.market.tz)
        # TODO: Report tz conversion

    return lob


def _normalize_index(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> pd.DataFrame:
    # Sort index
    if not lob.index.is_monotonic_increasing:
        lob = lob.sort_index()
        # TODO: Report LOB index ordering

    # Remove samples outside the market time
    assert isinstance(lob.index, pd.DatetimeIndex)
    mask = lob.index.indexer_between_time(asset.market.opening_time, asset.market.closing_time)
    n_outside = len(lob) - len(mask)
    if n_outside > 0:
        lob = lob.iloc[mask]
        # TODO: Report LOB time window adjustment

    # Remove index NaNs
    if lob.index.hasnans:
        mask = lob.index.isna().to_numpy()
        lob = lob[~mask]
        # TODO: Report LOB index remove nans

    # Remove index duplicates
    if lob.index.has_duplicates:
        mask = lob.index.duplicated()
        lob = lob[~mask]
        # TODO: Report LOB index remove dups

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
        # Report LOB index adjustment

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

    # TODO: Report missing columns
    # TODO: Report extra columns

    if missing or extra:
        lob = lob.reindex(columns=metadata.columns())

    return lob


def normalize(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
) -> pd.DataFrame:
    """"""
    lob = _normalize_timezone(lob, report, metadata, asset)
    lob = _normalize_index(lob, report, metadata, asset)
    lob = _normalize_columns(lob, report, metadata, asset)

    return lob


def _consistency_price_sign(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = metadata.columns(column_types="price")
    for _, s in lob[columns].items():
        s = s.dropna()
        mask = s <= 0
        if mask.any():
            # TODO: Report column, timestamps of the non-positive prices
            ...


def _consistency_price_unit(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = metadata.columns(column_types="price")
    for _, s in lob[columns].items():
        s = s.dropna()
        quotients = s / asset.tick_size_perc
        mask = ~np.isclose(quotients, np.round(quotients))
        if mask.any():
            # TODO: Report column, timestamps of the non-multiple prices
            ...


def _consistency_price_bid_order(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = [metadata.bid_price_column.format(level=level) for level in metadata.levels]
    for c1, c2 in zip(columns[:-1], columns[1:]):
        s1 = lob[c1]
        s2 = lob[c2]
        filter = s1.notna() & s2.notna()
        s1 = s1[filter]
        s2 = s2[filter]
        mask = s1 <= s2
        if mask.any():
            # TODO: Report columns, timestamps of non-ordered bid prices
            ...


def _consistency_price_ask_order(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = [metadata.ask_price_column.format(level=level) for level in metadata.levels]
    for c1, c2 in zip(columns[:-1], columns[1:]):
        s1 = lob[c1]
        s2 = lob[c2]
        filter = s1.notna() & s2.notna()
        s1 = s1[filter]
        s2 = s2[filter]
        mask = s1 >= s2
        if mask.any():
            # TODO: Report columns, timestamps of non-ordered ask prices
            ...


def _consistency_price_bid_ask_order(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    col_bid = metadata.bid_price_column.format(level=1)
    col_ask = metadata.ask_price_column.format(level=1)
    s_bid = lob[col_bid]
    s_ask = lob[col_ask]

    filter = s_bid.notna() & s_ask.notna()
    s_bid = s_bid[filter]
    s_ask = s_ask[filter]

    mask = s_bid >= s_ask
    if mask.any():
        # TODO: Best bid and ask unordred for amsk.sum() seconds
        ...


def _consistency_price(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    _consistency_price_sign(lob, report, metadata, asset)
    _consistency_price_unit(lob, report, metadata, asset)
    _consistency_price_bid_order(lob, report, metadata, asset)
    _consistency_price_ask_order(lob, report, metadata, asset)
    _consistency_price_bid_ask_order(lob, report, metadata, asset)
    ...


def _consistency_size_sign(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = metadata.columns(column_types="size")
    for _, s in lob[columns].items():
        s = s.dropna()
        mask = s <= 0
        if mask.any():
            # TODO: Report column, timestamps of the non-positive sizes
            ...


def _consistency_size_unit(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    columns = [c for c in lob.columns if "Size" in c]
    for _, s in lob[columns].items():
        s = s.dropna()
        mask = ~np.isclose(s, np.round(s))
        if mask.any():
            # TODO: Report column, timestamps of the non-multiple prices
            ...


def _consistency_size(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    _consistency_size_sign(lob, report, metadata, asset)
    _consistency_size_unit(lob, report, metadata, asset)
    ...


def consistency(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    """"""
    _consistency_price(lob, report, metadata, asset)
    _consistency_size(lob, report, metadata, asset)
    ...


def integrity(
    lob: pd.DataFrame, report: LobReport, metadata: config.LobMetadata, asset: config.AssetConfig
):
    """"""
    ...


def preprocess(path: Path, asset: config.AssetConfig, metadata: config.LobMetadata):
    """
    LOB preprocessing pipeline:
    * Load the parquet file to a DataFrame
    * Normalize to the expected LOB structure
    * Consistency check on the prices
    * Consistency check on the sizes

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
    lob = normalize(lob, report, metadata, asset)
    consistency(lob, report, metadata, asset)
    ...

    return lob, report
