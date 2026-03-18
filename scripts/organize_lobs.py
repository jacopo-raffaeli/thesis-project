"""Standardize and reorder raw LOB parquet files for futures and CTD.

Expected layout for both datasets:
timestamp (renamed to this exact name)
mid price column
For each level L: bid price, bid size, ask price, ask size

For futures only, if present, these columns are forced to the end:
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

    if MID_PRICE_COLUMN not in cols:
        raise ValueError(f"Expected mid price column '{MID_PRICE_COLUMN}' not found.")
    mid_price_col = MID_PRICE_COLUMN

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


def _process_file(src: Path, is_futures: bool) -> None:
    df = pd.read_parquet(src)
    ordered_cols, rename_map, index_rename = _build_ordered_columns(df, is_futures=is_futures)

    out_df = df.loc[:, ordered_cols].rename(columns=rename_map)

    # Rename index if needed
    if index_rename:
        out_df.index.name = index_rename

    out_df.to_parquet(src, engine="pyarrow")


def _collect_parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet")) if root.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorder and normalize raw parquet column layout.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("..") / "data" / "raw" / "FBTS",
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

    for src in ctd_files:
        if args.dry_run:
            print(f"[plan] CTD     {src} (replace in place)")
            continue
        _process_file(src=src, is_futures=False)
        print(f"[done] CTD     {src} (replaced)")

    for src in futures_files:
        if args.dry_run:
            print(f"[plan] FUTURES {src} (replace in place)")
            continue
        _process_file(src=src, is_futures=True)
        print(f"[done] FUTURES {src} (replaced)")

    if args.dry_run:
        print("Dry run complete. No files were modified.")
    else:
        print("Processing complete.")


if __name__ == "__main__":
    main()
