# TODO: Add warning statements where necessary
# TODO: Add minimal comments where needed

import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

_DATE_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})")
_DEFAULT_L1_COLUMNS = ("L1-BidPrice", "L1-BidSize", "L1-AskPrice", "L1-AskSize")
_PRICE_COL_PATTERN = re.compile(r"^L(\d+)-(Bid|Ask)Price$")
_SIZE_COL_PATTERN = re.compile(r"^L\d+-(BidSize|AskSize)$")


def _slice_intraday_window(
    lob_df: pd.DataFrame,
    start_time: Union[time, str],
    end_time: Union[time, str],
) -> pd.DataFrame:
    """Return intraday slice when index is datetime-based."""
    if not isinstance(lob_df.index, pd.DatetimeIndex):
        return lob_df
    start = _coerce_time(start_time)
    end = _coerce_time(end_time)
    return lob_df.between_time(start, end, inclusive="both")


def _coerce_time(value: Union[time, str]) -> time:
    """Convert HH:MM or HH:MM:SS strings to time objects."""
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).time()
            except ValueError:
                continue
        raise ValueError(f"Invalid intraday time value: {value}")
    raise ValueError(f"Unsupported intraday time type: {type(value)!r}")


def collect_timestamp_bounds_series(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
):
    """
    Collect first/last/last-non-nan timestamps for one file.

    TODO: Add variables description
    TODO: Add output description
    """
    mask_non_nan_row = lob_df.notna().any(axis=1)
    if not mask_non_nan_row.any():
        print(f"Warning: {filename} has rows but all values are NaN.")
        return None

    first_ts = pd.Timestamp(lob_df.index[0])
    last_ts = pd.Timestamp(lob_df.index[-1])
    last_non_nan_ts = pd.Timestamp(lob_df[mask_non_nan_row].index[-1])
    # Add other possible timestamps of interest

    # TODO: Index the output df by date
    return pd.DataFrame(
        [
            {
                "filename": filename,
                "lob_type": lob_type,
                "date": first_ts.date(),
                "first_timestamp": first_ts.time(),
                "last_timestamp": last_ts.time(),
                "last_non_nan_timestamp": last_non_nan_ts.time(),
            }
        ]
    )


def collect_nan_profile_series(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
):
    """
    Collect simple NaN profile statistics for one file.

    TODO: Add variables description
    TODO: Add output description
    """
    n_rows = len(lob_df)
    n_cols = len(lob_df.columns)
    n_cells = max(1, n_rows * n_cols)

    row_all_nan = lob_df.isna().all(axis=1)
    row_any_nan = lob_df.isna().any(axis=1)
    # col_nan_frac = lob_df.isna().mean().sort_values(ascending=False)

    # Create a string with the top 5 columns by NaN fraction
    # top_nan_cols = ",".join([f"{k}:{v:.3f}" for k, v in col_nan_frac.head(5).items()])

    first_ts = pd.Timestamp(lob_df.index[0])

    # TODO: Index the output df by date
    return pd.DataFrame(
        [
            {
                "filename": filename,
                "lob_type": lob_type,
                "date": first_ts.date(),
                "n_rows": n_rows,
                "n_cols": n_cols,
                "pct_rows_all_nan": float(row_all_nan.mean()),
                "pct_rows_any_nan": float(row_any_nan.mean()),
                "pct_nan_total": float(lob_df.isna().sum().sum() / n_cells),
                # "top_nan_columns": top_nan_cols,
            }
        ]
    )


def collect_sampling_gaps_series(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    expected_freq: str = "1s",
):
    """
    Collect missing-sample and gap metrics from the datetime index.

    TODO: Add variables description
    TODO: Add output description
    """
    if len(lob_df.index) < 2:
        first_ts = pd.Timestamp(lob_df.index[0])
        # TODO: Index the output df by date
        return pd.DataFrame(
            [
                {
                    "filename": filename,
                    "lob_type": lob_type,
                    "date": first_ts.date(),
                    "observed_points": len(lob_df),
                    "expected_points": len(lob_df),
                    "missing_points": 0,
                    "gap_count": 0,
                    "max_gap_seconds": 0,
                }
            ]
        )

    ordered_idx = pd.DatetimeIndex(lob_df.index).sort_values()
    full_idx = pd.date_range(start=ordered_idx.min(), end=ordered_idx.max(), freq=expected_freq)

    observed_points = len(ordered_idx)
    expected_points = len(full_idx)
    missing_points = max(0, expected_points - observed_points)

    delta_seconds = [
        int((ordered_idx[i].to_pydatetime() - ordered_idx[i - 1].to_pydatetime()).total_seconds())
        for i in range(1, len(ordered_idx))
    ]
    gap_over_1s = [d for d in delta_seconds if d > 1]

    # TODO: Index the output df by date
    return pd.DataFrame(
        [
            {
                "filename": filename,
                "lob_type": lob_type,
                "date": ordered_idx[0].date(),
                "observed_points": observed_points,
                "expected_points": expected_points,
                "missing_points": missing_points,
                "gap_count": int(len(gap_over_1s)),
                "max_gap_seconds": int(max(gap_over_1s)) if len(gap_over_1s) > 0 else 0,
            }
        ]
    )


def collect_ctd_opening_nan_diagnostics_series(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
):
    """
    Inspect opening NaN span in CTD files (returns None for futures files).

    TODO: Add variables description
    TODO: Add output description
    """
    if lob_type != "ctd":
        return None

    row_has_data = lob_df.notna().any(axis=1)
    first_ts = pd.Timestamp(lob_df.index[0])

    if not row_has_data.any():
        # TODO: Index the output df by date
        return pd.DataFrame(
            [
                {
                    "filename": filename,
                    "lob_type": lob_type,
                    "date": first_ts.date(),
                    "opening_rows_all_nan": len(lob_df),
                    "opening_nan_span_seconds": None,
                    "first_valid_timestamp": None,
                }
            ]
        )

    first_valid_ts = pd.Timestamp(lob_df[row_has_data].index[0])
    opening_rows_all_nan = int((~row_has_data).cumprod().sum())
    opening_nan_span = int((first_valid_ts - first_ts).total_seconds())

    # TODO: Index the output df by date
    return pd.DataFrame(
        [
            {
                "filename": filename,
                "lob_type": lob_type,
                "date": first_ts.date(),
                "opening_rows_all_nan": opening_rows_all_nan,
                "opening_nan_span_seconds": opening_nan_span,
                "first_valid_timestamp": first_valid_ts.time(),
            }
        ]
    )


def collect_l1_validity_jumps_series(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    l1_columns: Sequence[str] = _DEFAULT_L1_COLUMNS,
    start_time: Union[time, str] = time(9, 0),
    end_time: Union[time, str] = time(17, 0),
):
    """
    Collect event-level gaps between consecutive L1-valid samples.

    TODO: Add variables description
    TODO: Add output description
    """
    missing_cols = [col for col in l1_columns if col not in lob_df.columns]
    if missing_cols:
        print(f"Warning: {filename} is missing L1 columns: {missing_cols}")
        return None

    window_df = _slice_intraday_window(lob_df, start_time=start_time, end_time=end_time)
    if window_df.empty:
        return pd.DataFrame()

    start = _coerce_time(start_time)
    end = _coerce_time(end_time)

    valid_mask = pd.Series(
        window_df[list(l1_columns)].notna().to_numpy().all(axis=1),
        index=window_df.index,
    )
    if not valid_mask.any():
        return pd.DataFrame()

    valid_positions = [pd.Timestamp(ts) for ts in valid_mask[valid_mask].index]
    if len(valid_positions) < 2:
        return pd.DataFrame()

    first_ts = pd.Timestamp(window_df.index[0])
    n_rows_window = int(len(window_df))
    n_valid_rows_window = int(valid_mask.sum())
    pct_valid_rows_window = float(n_valid_rows_window / max(1, n_rows_window))

    records = []
    for i in range(1, len(valid_positions)):
        prev_valid_ts = valid_positions[i - 1]
        curr_valid_ts = valid_positions[i]

        jump_seconds = int((curr_valid_ts - prev_valid_ts).total_seconds())
        if jump_seconds <= 1:
            continue

        in_between = window_df.loc[
            (window_df.index > prev_valid_ts) & (window_df.index < curr_valid_ts)
        ]
        invalid_rows_between_valid = int(
            (
                ~pd.Series(
                    in_between[list(l1_columns)].notna().to_numpy().all(axis=1),
                    index=in_between.index,
                )
            ).sum()
        )

        records.append(
            {
                "filename": filename,
                "lob_type": lob_type,
                "date": first_ts.date(),
                "prev_valid_timestamp": prev_valid_ts.time(),
                "curr_valid_timestamp": curr_valid_ts.time(),
                "jump_seconds": jump_seconds,
                "missing_seconds_between_valid": max(0, jump_seconds - 1),
                "invalid_rows_between_valid": invalid_rows_between_valid,
                "window_start": start,
                "window_end": end,
                "n_rows_window": n_rows_window,
                "n_valid_rows_window": n_valid_rows_window,
                "pct_valid_rows_window": pct_valid_rows_window,
            }
        )

    return pd.DataFrame(records)


def collect_l1_validity_jump_stats_series(
    lob_df: pd.DataFrame,
    filename: str,
    lob_type: str,
    l1_columns: Sequence[str] = _DEFAULT_L1_COLUMNS,
    start_time: Union[time, str] = time(9, 0),
    end_time: Union[time, str] = time(17, 0),
):
    """Collect daily summary statistics for L1-validity jumps."""
    missing_cols = [col for col in l1_columns if col not in lob_df.columns]
    if missing_cols:
        print(f"Warning: {filename} is missing L1 columns: {missing_cols}")
        return None

    window_df = _slice_intraday_window(lob_df, start_time=start_time, end_time=end_time)
    if window_df.empty:
        return pd.DataFrame()

    start = _coerce_time(start_time)
    end = _coerce_time(end_time)

    valid_mask = pd.Series(
        window_df[list(l1_columns)].notna().to_numpy().all(axis=1),
        index=window_df.index,
    )
    first_ts = pd.Timestamp(window_df.index[0])

    valid_ts = [pd.Timestamp(ts) for ts in window_df.index[valid_mask]]
    jump_seconds = []
    if len(valid_ts) >= 2:
        jump_seconds = [
            int((valid_ts[i] - valid_ts[i - 1]).total_seconds())
            for i in range(1, len(valid_ts))
            if int((valid_ts[i] - valid_ts[i - 1]).total_seconds()) > 1
        ]

    invalid_mask = ~valid_mask
    max_consecutive_invalid_rows = 0
    if invalid_mask.any():
        grouped = invalid_mask.astype(int).groupby((~invalid_mask).cumsum()).sum()
        max_consecutive_invalid_rows = int(grouped.max())

    stats = {
        "filename": filename,
        "lob_type": lob_type,
        "date": first_ts.date(),
        "window_start": start,
        "window_end": end,
        "n_rows_window": int(len(window_df)),
        "n_valid_rows_window": int(valid_mask.sum()),
        "pct_valid_rows_window": float(valid_mask.mean()),
        "n_jumps_gt_1s": int(len(jump_seconds)),
        "mean_jump_seconds": float(pd.Series(jump_seconds).mean()) if jump_seconds else 0.0,
        "median_jump_seconds": float(pd.Series(jump_seconds).median()) if jump_seconds else 0.0,
        "p95_jump_seconds": float(pd.Series(jump_seconds).quantile(0.95)) if jump_seconds else 0.0,
        "max_jump_seconds": int(max(jump_seconds)) if jump_seconds else 0,
        "max_consecutive_invalid_rows": max_consecutive_invalid_rows,
    }

    return pd.DataFrame([stats])


def collect_missing_dates(
    base_path: Path, start_date: Optional[date] = None, end_date: Optional[date] = None
):
    """
    Return a DataFrame of dates missing in parquet filenames for one base path.

    TODO: Add variables description
    TODO: Add output description
    """
    # TODO: Search for Eurex/MTS official holiday calendars to compare the
    # missing dates
    observed_dates = []

    for file_path in sorted(base_path.rglob("*.parquet")):
        match = _DATE_PATTERN.search(file_path.name)
        if match is None:
            continue
        yyyy, mm, dd = match.groups()
        observed_dates.append(pd.Timestamp(year=int(yyyy), month=int(mm), day=int(dd)).date())

    if not observed_dates:
        return pd.DataFrame(columns=["missing_date"])

    min_date = min(observed_dates) if start_date is None else start_date
    max_date = max(observed_dates) if end_date is None else end_date

    all_dates = pd.date_range(min_date, max_date, freq="D").date
    missing = sorted(set(all_dates) - set(observed_dates))

    return pd.DataFrame({"missing_date": missing})


def collect_empty_raw_parquet_files(base_path: Path, lob_type: str) -> pd.DataFrame:
    """Return row counts and empty flags for all raw parquet files under a base path."""
    records = []

    for file_path in sorted(base_path.rglob("*.parquet")):
        date_value = None
        match = _DATE_PATTERN.search(file_path.name)
        if match is not None:
            yyyy, mm, dd = match.groups()
            date_value = pd.Timestamp(year=int(yyyy), month=int(mm), day=int(dd)).date()

        n_rows = None
        read_error = None
        try:
            df = pd.read_parquet(file_path, engine="fastparquet")
            n_rows = int(len(df))
        except Exception as exc:
            read_error = str(exc)

        records.append(
            {
                "filename": file_path.name,
                "relative_path": str(file_path.relative_to(base_path)).replace("\\", "/"),
                "lob_type": lob_type,
                "date": date_value,
                "n_rows": n_rows,
                "is_empty": bool(n_rows == 0) if n_rows is not None else None,
                "read_error": read_error,
            }
        )

    return pd.DataFrame(records)


def collect_negative_volume_series(lob_df: pd.DataFrame, filename: str, lob_type: str):
    """Collect per-file counts of strictly negative LOB sizes (NaNs are ignored)."""
    size_cols = [col for col in lob_df.columns if _SIZE_COL_PATTERN.match(col)]
    first_ts = pd.Timestamp(lob_df.index[0]) if len(lob_df.index) > 0 else pd.NaT

    if not size_cols:
        return pd.DataFrame(
            [
                {
                    "filename": filename,
                    "lob_type": lob_type,
                    "date": first_ts.date() if pd.notna(first_ts) else None,
                    "n_rows": int(len(lob_df)),
                    "n_size_columns": 0,
                    "n_negative_size_cells": 0,
                    "n_rows_with_negative_size": 0,
                    "has_negative_size": False,
                    "min_size_value": None,
                }
            ]
        )

    size_df = lob_df[size_cols].apply(pd.to_numeric, errors="coerce")
    negative_mask = size_df.to_numpy() < 0
    n_negative_cells = int(negative_mask.sum())
    n_rows_with_negative = int(negative_mask.any(axis=1).sum())
    min_size_value = float(size_df.min().min()) if not size_df.empty else None

    return pd.DataFrame(
        [
            {
                "filename": filename,
                "lob_type": lob_type,
                "date": first_ts.date() if pd.notna(first_ts) else None,
                "n_rows": int(len(lob_df)),
                "n_size_columns": int(len(size_cols)),
                "n_negative_size_cells": n_negative_cells,
                "n_rows_with_negative_size": n_rows_with_negative,
                "has_negative_size": bool(n_rows_with_negative > 0),
                "min_size_value": min_size_value,
            }
        ]
    )


def _price_level_map(columns: Iterable[str], side: str) -> dict[int, str]:
    level_map: dict[int, str] = {}
    for col in columns:
        match = _PRICE_COL_PATTERN.match(col)
        if match is None:
            continue
        level = int(match.group(1))
        col_side = match.group(2)
        if col_side == side:
            level_map[level] = col
    return dict(sorted(level_map.items()))


def _contiguous_valid_depth(row: pd.Series, level_map: dict[int, str]) -> int:
    depth = 0
    for level in sorted(level_map):
        value = row[level_map[level]]
        if pd.isna(value):
            break
        depth = level
    return depth


def collect_price_ordering_series(lob_df: pd.DataFrame, filename: str, lob_type: str):
    """Collect per-file row counts for bid/ask ladder and crossed-book ordering violations."""
    bid_map = _price_level_map(lob_df.columns, side="Bid")
    ask_map = _price_level_map(lob_df.columns, side="Ask")
    first_ts = pd.Timestamp(lob_df.index[0]) if len(lob_df.index) > 0 else pd.NaT

    n_bid_order_viol_rows = 0
    n_ask_order_viol_rows = 0
    n_crossed_book_rows = 0
    n_any_violation_rows = 0

    for _, row in lob_df.iterrows():
        bid_order_viol = False
        ask_order_viol = False
        crossed_viol = False

        bid_depth = _contiguous_valid_depth(row, bid_map)
        ask_depth = _contiguous_valid_depth(row, ask_map)

        if 1 in bid_map and 1 in ask_map:
            best_bid = row[bid_map[1]]
            best_ask = row[ask_map[1]]
            if pd.notna(best_bid) and pd.notna(best_ask) and float(best_bid) > float(best_ask):
                crossed_viol = True

        for level in range(2, bid_depth + 1):
            prev_value = row[bid_map[level - 1]]
            curr_value = row[bid_map[level]]
            if float(prev_value) < float(curr_value):
                bid_order_viol = True
                break

        for level in range(2, ask_depth + 1):
            prev_value = row[ask_map[level - 1]]
            curr_value = row[ask_map[level]]
            if float(prev_value) > float(curr_value):
                ask_order_viol = True
                break

        if bid_order_viol:
            n_bid_order_viol_rows += 1
        if ask_order_viol:
            n_ask_order_viol_rows += 1
        if crossed_viol:
            n_crossed_book_rows += 1
        if bid_order_viol or ask_order_viol or crossed_viol:
            n_any_violation_rows += 1

    return pd.DataFrame(
        [
            {
                "filename": filename,
                "lob_type": lob_type,
                "date": first_ts.date() if pd.notna(first_ts) else None,
                "n_rows": int(len(lob_df)),
                "n_bid_levels_available": int(len(bid_map)),
                "n_ask_levels_available": int(len(ask_map)),
                "n_bid_order_viol_rows": int(n_bid_order_viol_rows),
                "n_ask_order_viol_rows": int(n_ask_order_viol_rows),
                "n_crossed_book_rows": int(n_crossed_book_rows),
                "n_any_ordering_violation_rows": int(n_any_violation_rows),
                "has_ordering_violation": bool(n_any_violation_rows > 0),
            }
        ]
    )
