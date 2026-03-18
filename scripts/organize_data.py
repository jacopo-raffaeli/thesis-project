"""Organize cache folders: move files into ctd|futures/YYYY/MM/"""

import argparse
import re
from datetime import datetime
from pathlib import Path

DEFAULT_CACHES = ["FBTP", "FBTS"]
PAT = re.compile(r"(?P<type>ctd|futures).*?(?P<year>\d{4})[_-](?P<month>\d{2})", re.I)


def detect_type_and_date(name, fallback_mtime=None):
    m = PAT.search(name)
    if m:
        return m.group("type").lower(), int(m.group("year")), int(m.group("month"))
    if fallback_mtime:
        dt = datetime.fromtimestamp(fallback_mtime)
        return "unknown", dt.year, dt.month
    return "unknown", datetime.now().year, datetime.now().month


def main():
    ap = argparse.ArgumentParser(description="Organize cache_FBTP/cache_FBTS into type/YYYY/MM")
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
            if p.is_dir() or p.name.startswith("."):
                continue
            kind, year, month = detect_type_and_date(p.name, p.stat().st_mtime)
            if kind == "unknown":
                print(f"[skip] unknown type: {p.name}")
                continue

            target = cache_path / kind / f"{year:04d}" / f"{month:02d}" / p.name
            if args.dry_run or not args.execute:
                print(f"[plan] {p} -> {target}")
                continue

            # execute: move file, avoid overwrite by adding numeric suffix
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                p.replace(target)
                print(f"[moved] {p} -> {target}")
            else:
                base = target.stem
                suf = target.suffix
                i = 1
                while True:
                    cand = target.parent / f"{base}_{i}{suf}"
                    if not cand.exists():
                        p.replace(cand)
                        print(f"[moved] {p} -> {cand}")
                        break
                    i += 1


if __name__ == "__main__":
    main()
