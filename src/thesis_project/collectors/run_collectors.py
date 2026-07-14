import warnings
from datetime import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
)

import pandas as pd

from thesis_project import config as global_config

# Columns to drop from futures LOBs
_FUTURES_DROP_COLUMNS = global_config.FUT_COL_TO_DROP


def infer_lob_type(base_path: Path) -> str:
    """
    Infer LOB type from path name.

    TODO: Add variables description
    TODO: Add output description
    """
    if "futures" in base_path.name.lower():
        return "futures"
    elif "ctd" in base_path.name.lower():
        return "ctd"
    else:
        raise ValueError("Error: expected 'futures' or 'ctd' in the name.")


def iter_lob_parquets(base_path: Path, start_year: int, end_year: int) -> Iterator[Path]:
    """
    Yield parquet file paths in a year/month tree.

    # TODO: Add variables description
    # TODO: Add output description
    """
    for year_path in sorted(base_path.iterdir()):
        if not year_path.is_dir():
            warnings.warn(f"Skipping non-directory in year path: {year_path.name}", UserWarning)
            continue
        try:
            year_value = int(year_path.name)
        except ValueError:
            warnings.warn(f"Skipping non-numeric year folder: {year_path.name}", UserWarning)
            continue
        # Year cutoffs
        if year_value not in range(start_year, end_year + 1):
            continue
        for month_path in sorted(year_path.iterdir()):
            if not month_path.is_dir():
                warnings.warn(
                    f"Skipping non-directory in month path: {month_path.name}", UserWarning
                )
                continue
            for file_path in sorted(month_path.iterdir()):
                if file_path.is_file() and file_path.suffix == ".parquet":
                    yield file_path
                else:
                    warnings.warn(f"Skipping non-parquet file: {file_path.name}", UserWarning)
                    continue


def load_lob_dataframe(
    lob_path: Path, lob_type: str, start_time: time, end_time: time
) -> Optional[pd.DataFrame]:
    """
    Load one LOB parquet and apply preliminar cleaning steps.

    TODO: Add variables description
    TODO: Add output description
    """
    if lob_type not in {"futures", "ctd"}:
        raise ValueError("lob_type must be either 'futures' or 'ctd'")

    try:
        lob_df = pd.read_parquet(lob_path, engine="fastparquet")
    except Exception as exc:
        warnings.warn(f"Error reading {lob_path}: {exc}", UserWarning)
        return None

    # NOTE: Does it make sense, from the pov of analysis to return None?
    if lob_df.empty:
        warnings.warn(f"{lob_path} is empty.", UserWarning)
        return None

    # NOTE: Should I add a conditional cleaning or perform it anyway?
    # Drop useless columns from futures LOBs
    if lob_type == "futures":
        existing = [c for c in _FUTURES_DROP_COLUMNS if c in lob_df.columns]
        if existing:
            lob_df = lob_df.drop(columns=existing, errors="ignore")

    # NOTE: Should I add a conditional cleaning or perform it anyway?
    # Drop first row from ctd LOBs if all NaNs
    # This is a workaround for an issue with ctd lob generation
    if lob_type == "ctd" and lob_df.iloc[0].isna().all():
        lob_df = lob_df.iloc[1:]

    # NOTE: Should I add a conditional cutoff or perform it anyway?
    # Time cutoffs
    if isinstance(lob_df.index, pd.DatetimeIndex):
        lob_df = lob_df.between_time(start_time, end_time, inclusive="both")
    else:
        warnings.warn(
            f"{lob_path.name} index is not DatetimeIndex, time cutoff skipped.", UserWarning
        )

    return lob_df


def run_collectors_single(
    base_path: Path,
    lob_type: str,
    extractors: Dict[
        str, Callable[[pd.DataFrame, str, str, Optional[Dict]], Optional[pd.DataFrame]]
    ],
    start_year: int,
    end_year: int,
    start_time: time,
    end_time: time,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run multiple dataframe-based extractors in one pass over parquet files.

    TODO: Add variables description
    TODO: Add output description
    """

    if lob_type not in {"futures", "ctd"}:
        raise ValueError("lob_type must be either 'futures' or 'ctd'")

    records_by_extractor = {name: [] for name in extractors}

    for lob_path in iter_lob_parquets(base_path, start_year, end_year):
        lob_df = load_lob_dataframe(lob_path, lob_type, start_time, end_time)
        if lob_df is None:
            warnings.warn(f"Failed to load {lob_path.name}", UserWarning)
            continue

        for name, extractor in extractors.items():
            extracted_df = extractor(lob_df, lob_path.name, lob_type, context)
            if extracted_df is None or extracted_df.empty:
                warnings.warn(f"Extractor '{name}' returned empty for {lob_path.name}", UserWarning)
                continue
            records_by_extractor[name].append(extracted_df)

    result = {}
    for name, parts in records_by_extractor.items():
        if parts:
            df = pd.concat(parts, ignore_index=False).sort_index()
        else:
            df = pd.DataFrame()
        result[name] = df

    return result


def run_collectors_double(
    fut_base_path: Path,
    ctd_base_path: Path,
    extractors: Dict[
        str, Callable[[pd.DataFrame, pd.DataFrame, Optional[Dict]], Optional[pd.DataFrame]]
    ],
    start_year: int,
    end_year: int,
    start_time: time,
    end_time: time,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run multiple dataframe-based extractors in one pass over pairs of futures and CTD parquet files.

    Args:
        fut_base_path: Path to futures LOB data (year/month tree)
        ctd_base_path: Path to CTD LOB data (year/month tree)
        extractors: Dict mapping extractor names to callables with signature
                   (fut_df, ctd_df, context) -> Optional[pd.DataFrame]
        context: Optional dict with shared data for extractors (e.g., 'cf_series')
        start_year: Starting year for data collection
        end_year: Ending year for data collection
        start_time: Start time cutoff
        end_time: End time cutoff

    Returns:
        Dict mapping extractor names to concatenated DataFrames
    """
    if context is None:
        context = {}

    records_by_extractor = {name: [] for name in extractors}

    fut_paths = list(iter_lob_parquets(fut_base_path, start_year, end_year))
    ctd_paths = list(iter_lob_parquets(ctd_base_path, start_year, end_year))

    for fut_path, ctd_path in zip(fut_paths, ctd_paths):
        fut_lob_df = load_lob_dataframe(fut_path, "futures", start_time, end_time)
        ctd_lob_df = load_lob_dataframe(ctd_path, "ctd", start_time, end_time)

        if fut_lob_df is None:
            warnings.warn(f"Failed to load futures LOB from {fut_path.name}", UserWarning)
            continue

        if ctd_lob_df is None:
            warnings.warn(f"Failed to load CTD LOB from {ctd_path.name}", UserWarning)
            continue

        for name, extractor in extractors.items():
            extracted_df = extractor(fut_lob_df, ctd_lob_df, context)
            if extracted_df is None or extracted_df.empty:
                warnings.warn(
                    f"Extractor '{name}' returned empty for {fut_path.name}, {ctd_path.name}",
                    UserWarning,
                )
                continue
            records_by_extractor[name].append(extracted_df)

    result = {}
    for name, parts in records_by_extractor.items():
        if parts:
            df = pd.concat(parts, ignore_index=False).sort_index()
        else:
            df = pd.DataFrame()
        result[name] = df

    return result
