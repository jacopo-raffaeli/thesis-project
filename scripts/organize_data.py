"""
Organize lobs by month and year in ctd/YYYY/MM/ and futures/YYYY/MM/.
Both for FBTp and FBTS.
"""

import argparse
import re
from pathlib import Path

DEFAULT_CACHES = ["cache_FBTP", "cache_FBTS"]
PAT = re.compile(r"(?P<type>ctd|futures).*?(?P<year>\d{4})[_-](?P<month>\d{2})", re.I)

EXCLUDED_FILES = {"daily_cf.csv", "last_trading_dates.csv"}


def detect_type_and_date(name):
    m = PAT.search(name)
    if m:
        return m.group("type").lower(), int(m.group("year")), int(m.group("month"))
    raise ValueError(f"Cannot parse filename: {name}")


def main():
    ap = argparse.ArgumentParser(description="Organize cache_FBTP/cache_FBTS into lob_type/YYYY/MM")
    ap.add_argument(
        "--root",
        default="thesis-project/data/raw",
        help="root folder containing cache_FBTP and cache_FBTS",
    )
    ap.add_argument("--dry-run", action="store_true", help="only show planned moves")
    ap.add_argument("--execute", action="store_true", help="actually move files")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    for cache_name in DEFAULT_CACHES:
        cache_path = root / cache_name
        if not cache_path.exists():
            print(f"skip missing: {cache_path}")
            continue

        for p in sorted(cache_path.iterdir()):
            if p.is_dir() or p.name.startswith(".") or p.name in EXCLUDED_FILES:
                continue
            kind, year, month = detect_type_and_date(p.name)

            # Map cache_<LOB_TYPE> -> <lob_type>
            target_folder = cache_name.replace("cache_", "").lower()
            target = root / target_folder / kind / f"{year:04d}" / f"{month:02d}" / p.name
            if args.dry_run or not args.execute:
                if target.exists():
                    print(f"[plan] {p} -> {target} (replace)")
                else:
                    print(f"[plan] {p} -> {target}")
                continue

            # execute: move file, replace if exists
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                p.replace(target)
                print(f"[replaced] {p} -> {target}")
            else:
                p.replace(target)
                print(f"[moved] {p} -> {target}")


if __name__ == "__main__":
    main()
