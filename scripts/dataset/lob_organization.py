"""
Script to organize raw LOB data in folders.
The original data soruce is not modified.
Produce the following structure in the destionation folder: ticker/{ctd|fut}/yyyy/mm/{ctd|fut}_lob_freq_{freq}_yyyy_mm_dd.parquet

## Args:
* path: root path of raw LOB data, the expected data are the one that output "lob_pull.py"
* ticker: FutTicker
* mode: '--dry-run' or '--apply'
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

from thesis_project import config

logger = logging.getLogger(__name__)


LOB_FOLDER_MAP: dict[str, str] = {
    "ctd": "ctd",
    "futures": "fut",
}

FILENAME_RE = re.compile(
    r"(?P<role>ctd|futures).*?(?P<year>\d{4})[_-](?P<month>\d{2})",
    re.IGNORECASE,
)


def detect_role_and_date(name: str) -> tuple[str, int, int]:
    """Extract the LOB role, year, and month from a raw filename."""
    match = FILENAME_RE.search(name)

    if match is None:
        raise ValueError(f"Cannot parse LOB filename: {name}")

    role = match.group("role").lower()
    year = int(match.group("year"))
    month = int(match.group("month"))

    return role, year, month


def organize_lobs(
    *,
    src_root: Path,
    dst_root: Path,
    ticker: config.FutTicker,
    apply: bool,
) -> None:
    """
    Organize raw LOB files into the project's raw-data structure.

    Expected input:
        src_root/cache_<TICKER>/

    Output:
        dst_root/<ticker>/ctd/YYYY/MM/
        dst_root/<ticker>/fut/YYYY/MM/

    The only normalization performed here is the naming harmonization
    ``futures -> fut``. All other preprocessing is handled later.
    """
    src_root = src_root / f"cache_{ticker.upper()}"
    dst_root = dst_root / ticker

    if not src_root.exists():
        raise ValueError(f"Raw cache directory does not exist: {src_root}")

    if not src_root.is_dir():
        raise ValueError(f"Raw cache path is not a directory: {src_root}")

    logger.info("Source root: %s", src_root)
    logger.info("Destination root: %s", dst_root)

    files = sorted(
        path
        for path in src_root.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix == ".parquet"
    )

    copied = 0
    replaced = 0
    skipped = 0

    for src in files:
        try:
            role, year, month = detect_role_and_date(src.name)
        except ValueError:
            logger.warning("Skipping unrecognized file: %s", src)
            skipped += 1
            continue

        dst_role = LOB_FOLDER_MAP[role]

        filename = src.name
        if role == "futures":
            filename = filename.replace("futures", "fut", 1)

        dst = dst_root / dst_role / f"{year:04d}" / f"{month:02d}" / filename

        if dst.exists():
            action = "REPLACE"
            replaced += 1
        else:
            action = "COPY"
            copied += 1

        logger.info(
            "[%s] %s -> %s",
            "DRY-RUN" if not apply else action,
            src,
            dst,
        )

        if not apply:
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    total = copied + replaced + skipped

    logger.info("-" * 60)
    logger.info(
        "%s complete: %d files processed",
        "Dry run" if not apply else "Organization",
        total,
    )
    logger.info("  Copied:    %d", copied)
    logger.info("  Replaced: %d", replaced)
    logger.info("  Skipped:  %d", skipped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize raw LOB files into the project directory structure."
    )

    parser.add_argument(
        "src_root",
        type=Path,
        help="Absolute source path containing cache_<TICKER>.",
    )

    parser.add_argument(
        "dst_root",
        type=Path,
        help="Absolute destionation path.",
    )

    parser.add_argument(
        "--ticker",
        type=str,
        choices=config.get_args(config.FutTicker),
        help="Futures ticker.",
    )

    parser.add_argument(
        "--mode",
        choices=("dry-run", "apply"),
        default="dry-run",
        help="Whether to only show planned operations or actually copy files.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    organize_lobs(
        src_root=args.src_root,
        dst_root=args.dst_root,
        ticker=args.ticker,
        apply=args.mode == "apply",
    )


if __name__ == "__main__":
    main()
