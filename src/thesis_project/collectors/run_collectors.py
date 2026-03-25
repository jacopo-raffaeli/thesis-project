# TODO: Add warning statements where necessary
# TODO: Add minimal comments where needed

from datetime import time
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

import pandas as pd

_FUTURES_DROP_COLUMNS = ["#RIC", "Domain", "GMT Offset", "Type"]
# NOTE: The year cutoff is temporary, I will remove it once the whole dataset is
# available
_MAX_YEAR_INCLUDED = 2023
# NOTE: The time cutoff for futures is due to the fact that df extends up to
# 00:30 and we do not need data after 19:00, I will likely cut durectly the df
# in the future
_FUTURES_OPEN_TIME = time(8, 0)
_FUTURES_CLOSE_TIME = time(19, 0)


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


def iter_lob_parquets(base_path: Path) -> Iterator[Path]:
    """
    Yield parquet file paths in a year/month tree.

    TODO: Add variables description
    TODO: Add output description
    """
    # NOTE: For now force iteration to stop after 2023 since the parquets after
    # that are empty, do it both for futures and ctd, so that we have the same
    # cutoff for both datasets
    for year_path in sorted(base_path.iterdir()):
        if not year_path.is_dir():
            continue
        try:
            year_value = int(year_path.name)
        except ValueError:
            continue
        if year_value > _MAX_YEAR_INCLUDED:
            continue
        for month_path in sorted(year_path.iterdir()):
            if not month_path.is_dir():
                continue
            for file_path in sorted(month_path.iterdir()):
                if file_path.is_file() and file_path.suffix == ".parquet":
                    yield file_path


def run_collectors_multi(
    base_path: Path,
    extractors: Dict[str, Callable[[pd.DataFrame, str, str], Optional[pd.DataFrame]]],
) -> Dict[str, pd.DataFrame]:
    """
    Run multiple dataframe-based extractors in one pass over parquet files.

    TODO: Add variables description
    TODO: Add output description
    """
    lob_type = infer_lob_type(base_path)
    records_by_extractor = {name: [] for name in extractors}

    for lob_path in iter_lob_parquets(base_path):
        lob_df = load_lob_dataframe(lob_path, lob_type)
        if lob_df is None:
            continue

        for name, extractor in extractors.items():
            extracted_df = extractor(lob_df, lob_path.name, lob_type)
            if extracted_df is None or extracted_df.empty:
                continue
            records_by_extractor[name].append(extracted_df)

    # TODO: Is it a problem to ignore index here if we want to index by date?
    return {
        name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for name, parts in records_by_extractor.items()
    }


def load_lob_dataframe(lob_path: Path, lob_type: str) -> Optional[pd.DataFrame]:
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
        print(f"Error reading {lob_path}: {exc}")
        return None

    if lob_df.empty:
        # print(f"Warning: {lob_path} is empty.")
        return None

    if lob_type == "futures":
        existing = [c for c in _FUTURES_DROP_COLUMNS if c in lob_df.columns]
        if existing:
            lob_df = lob_df.drop(columns=existing, errors="ignore")

        if isinstance(lob_df.index, pd.DatetimeIndex):
            lob_df = lob_df.between_time(_FUTURES_OPEN_TIME, _FUTURES_CLOSE_TIME, inclusive="both")
        else:
            print(f"Warning: {lob_path.name} index is not DatetimeIndex; time cutoff skipped.")

    return lob_df
