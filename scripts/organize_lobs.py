"""
Standardize and reorder raw LOB parquet files for futures and CTD.

Expected input layout for both datasets:
'timestamp' and 'Date-Time' respectively for ctd and futures
'mid' column for mid price
'L<n>-<Bid|Ask><Price|Size>' columns for n=1..10
'#RIC', 'Domain', 'GMT Offset', 'Type' columns for futures only

Expected output layout for both datasets:
'timestamp' is the index
'MidPrice' column
'L<n>-<Bid|Ask><Price|Size>' columns for n=1..10

For futures only, if present, these columns are forced to the end:
- #RIC
- Domain
- GMT Offset
- Type
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CTD_TIMESTAMP_COLUMN = "timestamp"
FUTURES_TIMESTAMP_COLUMN = "Date-Time"
MID_PRICE_COLUMN = "mid"

FUTURES_TAIL_COLUMNS = ("#RIC", "Domain", "GMT Offset", "Type")


def _find_column_case_insensitive(cols: list[str], target: str) -> str | None:
    """
    Find a column name case-insensitively. Returns the actual column name or None.
    """
    target_lower = target.lower()
    for col in cols:
        if col.lower() == target_lower:
            return col
    return None


def _build_lob_columns() -> list[str]:
    cols: list[str] = []
    for i in range(1, 11):
        cols.extend(
            [
                f"L{i}-BidPrice",
                f"L{i}-BidSize",
                f"L{i}-AskPrice",
                f"L{i}-AskSize",
            ]
        )
    return cols


def _build_ordered_columns(
    df: pd.DataFrame, is_futures: bool
) -> tuple[list[str], dict[str, str], str | None]:
    cols = list(df.columns)

    # Determine expected timestamp and verify it's in the index
    timestamp_col = FUTURES_TIMESTAMP_COLUMN if is_futures else CTD_TIMESTAMP_COLUMN
    if df.index.name != timestamp_col:
        raise ValueError(
            f"Expected timestamp index '{timestamp_col}' not found. Found: {df.index.name}"
        )

    # Find mid price column (case-insensitive)
    mid_price_col = _find_column_case_insensitive(cols, MID_PRICE_COLUMN)
    if mid_price_col is None:
        raise ValueError(
            f"Expected mid price column '{MID_PRICE_COLUMN}' not found. Available: {cols}"
        )

    lob_cols = [c for c in _build_lob_columns() if c in cols]

    ordered: list[str] = [mid_price_col] + lob_cols

    futures_tail = [c for c in FUTURES_TAIL_COLUMNS if c in cols] if is_futures else []

    # Keep any remaining columns without dropping information.
    ordered_set = set(ordered)
    tail_set = set(futures_tail)
    remaining = [c for c in cols if c not in ordered_set and c not in tail_set]

    final_order = ordered + remaining + futures_tail
    rename_map = {}
    if mid_price_col != "MidPrice":
        rename_map[mid_price_col] = "MidPrice"

    # Return index rename map (rename the index itself, not as a column)
    index_rename = None
    if timestamp_col != "timestamp":
        index_rename = "timestamp"

    return final_order, rename_map, index_rename


def _process_file(src: Path, is_futures: bool) -> tuple[bool, str]:
    """Process a single parquet file. Returns (success, message)."""
    try:
        # Read parquet
        try:
            df = pd.read_parquet(src)
        except Exception as e:
            return False, f"Failed to read parquet: {type(e).__name__}: {e}"

        # Check if empty
        if df.empty:
            return False, "File is empty (no rows)"

        # Validate and build ordered columns
        try:
            ordered_cols, rename_map, index_rename = _build_ordered_columns(
                df, is_futures=is_futures
            )
        except ValueError as e:
            return False, f"Column validation failed: {e}"

        # Transform dataframe
        try:
            out_df = df.loc[:, ordered_cols].rename(columns=rename_map)
            if index_rename:
                out_df.index.name = index_rename
        except Exception as e:
            return False, f"Failed to transform dataframe: {type(e).__name__}: {e}"

        # Write parquet
        try:
            out_df.to_parquet(src, engine="pyarrow")
        except Exception as e:
            return False, f"Failed to write parquet: {type(e).__name__}: {e}"

        # Verify written file
        try:
            verify_df = pd.read_parquet(src)
            if verify_df.empty:
                return False, "Verification failed: written file is empty"
            if len(verify_df) != len(out_df):
                return (
                    False,
                    f"Verification failed: row count mismatch (wrote {len(out_df)}, read {len(verify_df)})",
                )
            if list(verify_df.columns) != list(out_df.columns):
                return False, "Verification failed: column mismatch"
        except Exception as e:
            return False, f"Verification failed: {type(e).__name__}: {e}"

        return True, "Success"

    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {e}"


def _collect_parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet")) if root.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorder and normalize raw parquet column layout.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("..") / "data" / "raw" / "fbtp",
        help="Root folder containing ctd/ and futures/.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print files that would be processed."
    )
    args = parser.parse_args()

    input_root = args.input_root

    ctd_root = input_root / "ctd"
    futures_root = input_root / "futures"

    ctd_files = _collect_parquet_files(ctd_root)
    futures_files = _collect_parquet_files(futures_root)

    total = len(ctd_files) + len(futures_files)
    if total == 0:
        raise SystemExit(f"No parquet files found under: {input_root}")

    print(f"Found {len(ctd_files)} CTD files and {len(futures_files)} futures files.")

    success_count = 0
    fail_count = 0

    for src in ctd_files:
        if args.dry_run:
            print(f"[plan] CTD     {src}")
            continue
        success, message = _process_file(src=src, is_futures=False)
        if success:
            print(f"[done] CTD     {src}")
            success_count += 1
        else:
            print(f"[fail] CTD     {src}")
            print(f"       Error: {message}")
            fail_count += 1

    for src in futures_files:
        if args.dry_run:
            print(f"[plan] FUTURES {src}")
            continue
        success, message = _process_file(src=src, is_futures=True)
        if success:
            print(f"[done] FUTURES {src}")
            success_count += 1
        else:
            print(f"[fail] FUTURES {src}")
            print(f"       Error: {message}")
            fail_count += 1

    if args.dry_run:
        print("Dry run complete. No files were modified.")
    else:
        print(f"\nProcessing complete: {success_count} succeeded, {fail_count} failed.")
        if fail_count > 0:
            raise SystemExit(f"Processing failed for {fail_count} file(s). See errors above.")


if __name__ == "__main__":
    main()
