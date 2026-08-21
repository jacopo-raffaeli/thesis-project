import warnings
from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from thesis_project import config as global_config
from thesis_project.utils.io import filename_to_date


def collect_empty_lobs(
    df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Analyze if a lob is empty or not and return a dataframe with the results.

    Args:
        df: DataFrame with LOB data
        filename: Name of the file being analyzed
        lob_type: Type of the LOB ("futures" or "ctd")

    Returns:
        A DataFrame with the filename, lob_type, date, and empty flag.
    """
    date = filename_to_date(filename)
    date = pd.to_datetime(date) if pd.notna(date) else pd.NaT

    out_df = pd.DataFrame(
        {
            "Filename": filename,
            "LOB type": lob_type,
            "Empty": df.empty,
        },
        index=[date],
    )
    out_df.index.name = "Date"

    return out_df


def collect_nans(
    df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Collect missing elements in the lob and return a dataframe with the results.

    Args:
        df: DataFrame with LOB data
        filename: Name of the file being analyzed
        lob_type: Type of the LOB ("futures" or "ctd")

    Returns:
        A DataFrame with the filename, lob_type and NaNs statistics.
    """
    date = filename_to_date(filename)
    date = pd.to_datetime(date) if pd.notna(date) else pd.NaT

    # Drop MidPrice since it is derived from other prices columns
    df = df.drop(columns=["MidPrice"], errors="ignore")
    n_row, n_col = df.shape
    na = df.isna()

    # Row stats
    row_all_nan = na.all(axis=1)
    row_any_nan = na.any(axis=1)
    n_row_all_nan = row_all_nan.sum()
    n_row_any_nan = row_any_nan.sum()

    # Column stats
    col_all_nan = na.all(axis=0)
    col_any_nan = na.any(axis=0)
    n_col_all_nan = col_all_nan.sum()
    n_col_any_nan = col_any_nan.sum()

    # Total NaNs
    n_nan = na.sum().sum()
    total_cells = n_row * n_col
    n_nan_pct = n_nan / total_cells if total_cells > 0 else 0.0

    out_df = pd.DataFrame(
        {
            "Filename": filename,
            "LOB type": lob_type,
            "Rows": n_row,
            "Rows all NaNs": n_row_all_nan,
            "Rows all NaNs (%)": (n_row_all_nan / n_row * 100) if n_row else 0,
            "Rows any NaNs": n_row_any_nan,
            "Rows any NaNs (%)": (n_row_any_nan / n_row * 100) if n_row else 0,
            "Cols": n_col,
            "Cols all NaNs": n_col_all_nan,
            "Cols all NaNs (%)": (n_col_all_nan / n_col * 100) if n_col else 0,
            "Cols any NaNs": n_col_any_nan,
            "Cols any NaNs (%)": (n_col_any_nan / n_col * 100) if n_col else 0,
            "Total NaNs": n_nan,
            "Total NaNs (%)": n_nan_pct * 100,
        },
        index=[date],
    )
    out_df.index.name = "Date"

    return out_df


def collect_timestamps(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
):
    """
    Collect relevant timestamps in the lov and return a dataframe with the results.

    Args:
        df: DataFrame with LOB data
        filename: Name of the file being analyzed
        lob_type: Type of the LOB ("futures" or "ctd")

    Returns:
        A DataFrame with the filename, lob_type and relevant timestamps.
    """
    date = filename_to_date(filename)
    date = pd.to_datetime(date) if pd.notna(date) else pd.NaT

    mask_non_nan_row = lob_df.notna().any(axis=1)
    if not mask_non_nan_row.any():
        print(f"Warning: {filename} has rows but all values are NaN.")
        return None

    first_ts = lob_df.index[0].time()
    last_ts = lob_df.index[-1].time()
    first_non_nan_ts = lob_df[mask_non_nan_row].index[0].time()
    last_non_nan_ts = lob_df[mask_non_nan_row].index[-1].time()
    # Add other possible timestamps of interest

    out_df = pd.DataFrame(
        {
            "Filename": filename,
            "LOB type": lob_type,
            "First timestamp": first_ts,
            "First non-empty timestamp": first_non_nan_ts,
            "Last non-empty timestamp": last_non_nan_ts,
            "Last timestamp": last_ts,
        },
        index=[date],
    )
    out_df.index.name = "Date"

    return out_df


def collect_fut_integrity(
    df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Analyze futures specific integrity issues

    Args:
        df: DataFrame with LOB data
        filename: Name of the file being analyzed
        lob_type: Type of the LOB ("futures" or "ctd")

    Returns:
        A DataFrame with the filename, lob_type and futures-specific integrity metrics.
    """
    date = filename_to_date(filename)
    date = pd.to_datetime(date) if pd.notna(date) else pd.NaT

    if lob_type != "futures":
        warnings.warn(
            f"Expected lob_type 'futures' for collect_fut_integrity, got '{lob_type}'.", UserWarning
        )
        return None

    min = time(0, 1, 0)
    max = time(19, 0, 0)
    cutoff = (df.index.time < min) | (df.index.time > max)  # type: ignore
    df = df.loc[cutoff]

    if not df.empty:
        nans_df = collect_nans(df, filename, lob_type)
    else:
        nans_df = collect_nans(pd.DataFrame(), filename, lob_type)

    nans_df = nans_df.drop(columns=["Filename", "LOB type"])  # type: ignore

    out_df = pd.DataFrame(
        {
            "Filename": filename,
            "LOB type": lob_type,
            "First timestamp": df.index[0].time() if not df.empty else pd.NaT,
            "Last timestamp": df.index[-1].time() if not df.empty else pd.NaT,
        },
        index=[date],
    )
    out_df.index.name = "Date"

    out_df = pd.concat([out_df, nans_df], axis=1)

    return out_df


def collect_ctd_integrity(
    df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Analyze CTD specific integrity issues

    Args:
        df: DataFrame with LOB data
        filename: Name of the file being analyzed
        lob_type: Type of the LOB ("futures" or "ctd")

    Returns:
        A DataFrame with the filename, lob_type and CTD-specific integrity metrics.
    """
    date = filename_to_date(filename)
    date = pd.to_datetime(date) if pd.notna(date) else pd.NaT

    if lob_type != "ctd":
        warnings.warn(
            f"Expected lob_type 'ctd' for collect_ctd_integrity, got '{lob_type}'.", UserWarning
        )
        return None

    out_df = pd.DataFrame(
        {
            "Filename": filename,
            "LOB type": lob_type,
            "First timestamp empty": df.iloc[0].isna().all(),
        },
        index=[date],
    )
    out_df.index.name = "Date"

    return out_df


def collect_spread_sign(
    df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Analyze bid-ask spread consistency sign for all levels of the LOB

    Args:
        df: DataFrame with LOB data
        filename: Name of the file being analyzed
        lob_type: Type of the LOB ("futures" or "ctd")

    Returns:
        A DataFrame with the filename, lob_type and spread sign consistency results.
    """
    date = filename_to_date(filename)
    date = pd.to_datetime(date) if pd.notna(date) else pd.NaT

    out_df = pd.DataFrame(
        {
            "Filename": filename,
            "LOB type": lob_type,
        },
        index=[date],
    )
    out_df.index.name = "Date"

    bid_cols = [c for c in df.columns if global_config.BID_PRICE_COL_PATTERN.match(c)]
    ask_cols = [c for c in df.columns if global_config.ASK_PRICE_COL_PATTERN.match(c)]

    spread_check = {
        i: bool(((df[ask_col] - df[bid_col]).dropna() >= 0).all())
        for i, (bid_col, ask_col) in enumerate(zip(bid_cols, ask_cols), start=1)
    }

    for i in spread_check:
        out_df[f"L{i} spread check"] = spread_check[i]

    out_df["Overall spread check"] = bool(all(spread_check.values()))

    return out_df


def collect_volume_sign(
    df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Analyze volume sign consistency for all levels of the LOB

    Args:
        df: DataFrame with LOB data
        filename: Name of the file being analyzed
        lob_type: Type of the LOB ("futures" or "ctd")

    Returns:
        A DataFrame with the filename, lob_type and volume sign consistency results.
    """
    date = filename_to_date(filename)
    date = pd.to_datetime(date) if pd.notna(date) else pd.NaT

    out_df = pd.DataFrame(
        {
            "Filename": filename,
            "LOB type": lob_type,
        },
        index=[date],
    )
    out_df.index.name = "Date"

    bid_cols = [c for c in df.columns if global_config.BID_SIZE_COL_PATTERN.match(c)]
    ask_cols = [c for c in df.columns if global_config.ASK_SIZE_COL_PATTERN.match(c)]

    bid_check = {i: bool((df[col].dropna() >= 0).all()) for i, col in enumerate(bid_cols, start=1)}
    ask_check = {i: bool((df[col].dropna() >= 0).all()) for i, col in enumerate(ask_cols, start=1)}

    for i in bid_check:
        out_df[f"L{i} bid volume check"] = bid_check[i]
    out_df["Overall bid volume check"] = bool(all(bid_check.values()))
    for i in ask_check:
        out_df[f"L{i} ask volume check"] = ask_check[i]
    out_df["Overall ask volume check"] = bool(all(ask_check.values()))

    out_df["Overall volume check"] = bool(all(bid_check.values()) and all(ask_check.values()))

    return out_df


def collect_bid_ask_order(
    df: pd.DataFrame,
    filename: str,
    lob_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Analyze bid and ask price ordering consistency for all levels of the LOB

    Args:
        df: DataFrame with LOB data
        filename: Name of the file being analyzed
        lob_type: Type of the LOB ("futures" or "ctd")

    Returns:
        A DataFrame with the filename, lob_type and bid-ask ordering consistency results.
    """
    date = filename_to_date(filename)
    date = pd.to_datetime(date) if pd.notna(date) else pd.NaT

    out_df = pd.DataFrame(
        {
            "Filename": filename,
            "LOB type": lob_type,
        },
        index=[date],
    )
    out_df.index.name = "Date"

    bid_cols = [c for c in df.columns if global_config.BID_PRICE_COL_PATTERN.match(c)]
    ask_cols = [c for c in df.columns if global_config.ASK_PRICE_COL_PATTERN.match(c)]

    bid_df = df[bid_cols]
    ask_df = df[ask_cols]

    bid_check = ((bid_df.diff(axis=1) <= 0) | bid_df.diff(axis=1).isna()).all(axis=1)
    ask_check = ((ask_df.diff(axis=1) >= 0) | ask_df.diff(axis=1).isna()).all(axis=1)

    out_df["Bid price decreasing ordering check"] = bool(bid_check.all())
    out_df["Ask price increasing ordering check"] = bool(ask_check.all())
    out_df["Overall price order check"] = bool(bid_check.all() and ask_check.all())

    return out_df


# def collect_missing_dates(
#     base_path: Path, start_date: Optional[date] = None, end_date: Optional[date] = None
# ):
#     """
#     Return a DataFrame of dates missing in parquet filenames for one base path.

#     TODO: Add variables description
#     TODO: Add output description
#     """
#     # TODO: Search for Eurex/MTS official holiday calendars to compare the
#     # missing dates
#     observed_dates = []

#     for file_path in sorted(base_path.rglob("*.parquet")):
#         match = config.DATE_PATTERN.search(file_path.name)
#         if match is None:
#             continue
#         yyyy, mm, dd = match.groups()
#         observed_dates.append(pd.Timestamp(year=int(yyyy), month=int(mm), day=int(dd)).date())

#     if not observed_dates:
#         return pd.DataFrame(columns=["missing_date"])

#     min_date = min(observed_dates) if start_date is None else start_date
#     max_date = max(observed_dates) if end_date is None else end_date

#     all_dates = pd.date_range(min_date, max_date, freq="D").date
#     missing = sorted(set(all_dates) - set(observed_dates))

#     return pd.DataFrame({"missing_date": missing})
