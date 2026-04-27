"""Snapshot OHLCV from yfinance into ``data/snapshots/`` for reproducibility.

Closes audit issue C-13. Snapshots are Parquet files versioned by
ticker and date range. Once committed, every downstream consumer
(:mod:`src.data.download_stock_data`,
:func:`src.pipeline.run_nested_purged_cv`,
``scripts/run_backtest.py``) reads the cached file by default and
hits yfinance only when the snapshot is missing.

Usage
-----

Snapshot a single ticker at the project's standard 2020-2024 window::

    python scripts/snapshot_data.py --tickers AAPL

Snapshot the full panel from ``src.config.STOCK_UNIVERSE``::

    python scripts/snapshot_data.py

Force a refresh even if a snapshot exists::

    python scripts/snapshot_data.py --tickers AAPL --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import CONFIG, get_all_tickers  # noqa: E402
from src.data import _snapshot_path, save_snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data snapshot writer")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Tickers to snapshot (default: full panel from config)")
    p.add_argument("--start", default=CONFIG["start_date"])
    p.add_argument("--end", default=CONFIG["end_date"])
    p.add_argument("--force", action="store_true",
                   help="Refresh snapshots even if they already exist")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tickers = args.tickers or get_all_tickers()

    print("=" * 72)
    print(" Data snapshot")
    print("=" * 72)
    print(f" tickers : {tickers}")
    print(f" range   : {args.start} -> {args.end}")
    print(f" force   : {args.force}")
    print()

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for ticker in tickers:
        path = _snapshot_path(ticker, args.start, args.end)
        if path.exists() and not args.force:
            print(f"  [skip] {ticker} -> {path.name} (already exists)")
            n_skipped += 1
            continue
        result = save_snapshot(ticker, args.start, args.end)
        if result is None:
            print(f"  [fail] {ticker}")
            n_failed += 1
        else:
            n_done += 1

    print()
    print(f"Wrote: {n_done}  Skipped: {n_skipped}  Failed: {n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
