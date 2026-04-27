"""Driver for the economic backtest of leakage-safe OOF predictions.

Reads ``results/nested_cv_predictions.csv`` (and the per-horizon
``results/nested_cv_h{N}/nested_cv_predictions.csv``), refetches AAPL
OHLCV via yfinance to obtain exit prices, runs the strategy at every
``(model, cost_bps)`` cell, and writes:

  results/backtest_metrics.csv  — one row per (ticker, model, horizon, cost, strategy)
  results/backtest_equity.csv   — long-form equity curves
  results/backtest_run_snapshot.json

Usage::

    python scripts/run_backtest.py            # backtest every horizon found in results/
    python scripts/run_backtest.py --horizons 1
    python scripts/run_backtest.py --costs-bps 0 5 10 25 --strategy sign
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.backtest import sweep_backtests  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"


def _predictions_path_for(horizon: int) -> Path:
    """Locate the OOF predictions CSV for a given horizon."""
    if horizon == 1:
        # h=1 is in the top-level results/ (matches the qf-04 layout).
        return RESULTS_DIR / "nested_cv_predictions.csv"
    return RESULTS_DIR / f"nested_cv_h{horizon}" / "nested_cv_predictions.csv"


def _load_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch OHLCV via yfinance and return the close series."""
    from src.data import download_stock_data
    df = download_stock_data(ticker, start, end, verbose=False)
    if df is None or df.empty:
        raise RuntimeError(f"No price data fetched for {ticker}")
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Economic backtest runner")
    p.add_argument("--horizons", nargs="*", type=int, default=None,
                   help="Horizons to backtest (default: 1, 5, 15 if files exist)")
    p.add_argument("--costs-bps", nargs="*", type=float,
                   default=[0.0, 5.0, 10.0, 25.0],
                   help="Round-trip transaction costs in bps")
    p.add_argument("--strategy", choices=["sign", "threshold", "both"],
                   default="sign")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Conviction threshold for the threshold strategy "
                        "(applied only when --strategy in {threshold,both})")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Discover available horizons if none specified.
    horizons = args.horizons
    if horizons is None:
        horizons = []
        for h in (1, 2, 5, 10, 15):
            if _predictions_path_for(h).exists():
                horizons.append(h)
    if not horizons:
        print("No predictions found. Run scripts/run_nested_cv.py first.",
              file=sys.stderr)
        return 1

    if args.strategy == "both":
        strategies = ("sign", "threshold")
    else:
        strategies = (args.strategy,)

    print("=" * 72)
    print(" Economic backtest")
    print("=" * 72)
    print(f" horizons        : {horizons}")
    print(f" costs (bps)     : {args.costs_bps}")
    print(f" strategies      : {strategies}")
    print(f" threshold       : {args.threshold}")
    print()

    all_metrics: list[pd.DataFrame] = []
    all_equity:  list[pd.DataFrame] = []

    for h in horizons:
        preds_path = _predictions_path_for(h)
        print(f"[h={h}] loading {preds_path.relative_to(REPO_ROOT)}")
        preds = pd.read_csv(preds_path, parse_dates=["date"])
        # Restrict to AAPL — predictions today only cover AAPL.
        for ticker, sub_t in preds.groupby("ticker"):
            print(f"  ticker={ticker} n_pred={len(sub_t)} models={sub_t['model'].unique().tolist()}")
            prices = _load_prices(ticker, args.start, args.end)
            metrics, equity = sweep_backtests(
                sub_t, prices,
                horizon=h,
                cost_bps_grid=args.costs_bps,
                strategies=strategies,
                threshold=args.threshold,
                ticker=ticker,
            )
            all_metrics.append(metrics)
            all_equity.append(equity)

    if not all_metrics:
        print("No results produced.", file=sys.stderr)
        return 1

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    equity_df = pd.concat(all_equity, ignore_index=True)

    metrics_path = results_dir / "backtest_metrics.csv"
    equity_path = results_dir / "backtest_equity.csv"
    snap_path = results_dir / "backtest_run_snapshot.json"

    metrics_df.to_csv(metrics_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    snap = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "horizons": horizons,
        "costs_bps": args.costs_bps,
        "strategies": list(strategies),
        "threshold": args.threshold,
        "start": args.start,
        "end": args.end,
        "n_metric_rows": int(len(metrics_df)),
        "n_equity_rows": int(len(equity_df)),
    }
    snap_path.write_text(json.dumps(snap, indent=2))

    # Pretty summary
    print()
    print(f"Wrote {metrics_path}")
    print(f"Wrote {equity_path}")
    print(f"Wrote {snap_path}")
    print()
    summary_cols = [
        "ticker", "model", "horizon", "cost_bps", "strategy",
        "n_trades", "hit_rate",
        "gross_ann_return", "net_ann_return", "net_sharpe",
        "max_drawdown", "buy_and_hold_ann_return",
    ]
    print("=== Headline backtest summary ===")
    print(metrics_df[summary_cols].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
