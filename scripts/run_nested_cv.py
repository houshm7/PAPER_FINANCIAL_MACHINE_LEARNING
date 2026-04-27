"""Driver for the leakage-safe nested Purged K-Fold pipeline.

Examples
--------

Smoke test with synthetic data (no yfinance, fully deterministic, ~30s)::

    python scripts/run_nested_cv.py --synthetic --smoke

Real-data smoke (1 stock, small settings, ~2-5 min)::

    python scripts/run_nested_cv.py --smoke --tickers AAPL

Full panel run (multi-hour on CPU; this is the headline-paper
configuration; only run when explicitly authorised)::

    python scripts/run_nested_cv.py

Outputs four CSVs in ``results/``:

- ``nested_cv_predictions.csv``      — OOF predictions, one row per (date, model)
- ``nested_cv_metrics.csv``          — per-fold + pooled metrics
- ``nested_cv_selected_features.csv``— corr/boruta selections per fold
- ``nested_cv_best_params.csv``      — Optuna best params per fold per model

plus a ``nested_cv_run_snapshot.json`` with the full config and seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import CONFIG, get_all_tickers  # noqa: E402
from src.pipeline import run_nested_purged_cv  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Synthetic OHLCV (deterministic, used by --synthetic to avoid yfinance)
# ---------------------------------------------------------------------------

def synthetic_ohlcv(ticker: str, n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Reproducible OHLCV with a mild momentum signal so models can learn
    *something*. Index = trading-day calendar starting 2020-01-01."""
    rng = np.random.default_rng(seed + abs(hash(ticker)) % 1_000_000)
    drift = 0.0005
    rets = rng.normal(loc=drift, scale=0.012, size=n)
    # Inject a small AR(1) component → next-day momentum.
    for t in range(1, n):
        rets[t] += 0.10 * rets[t - 1]
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0, 0.01, size=n))
    low = close * (1.0 - rng.uniform(0, 0.01, size=n))
    open_ = close * (1.0 + rng.normal(0, 0.002, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested Purged K-Fold runner")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny settings (3x2 folds, 2 trials, 2 models) for verification")
    p.add_argument("--synthetic", action="store_true",
                   help="Use deterministic synthetic OHLCV (no yfinance)")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Override ticker list (default: all 25 from config)")
    p.add_argument("--window", type=int, default=1,
                   help="Forecasting horizon h (default: 1)")
    p.add_argument("--label-mode", default="raw_return",
                   help='Default "raw_return" (causal). Pass "wavelet" only as robustness.')
    p.add_argument("--n-outer", type=int, default=None)
    p.add_argument("--n-inner", type=int, default=None)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--model-names", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict:
    if args.smoke:
        defaults = dict(
            tickers=args.tickers or ["AAPL"],
            n_outer=args.n_outer or 3,
            n_inner=args.n_inner or 2,
            n_trials=args.n_trials or 2,
            model_names=args.model_names or ["XGBoost", "Random Forest"],
        )
    else:
        defaults = dict(
            tickers=args.tickers or get_all_tickers(),
            n_outer=args.n_outer or 5,
            n_inner=args.n_inner or 5,
            n_trials=args.n_trials or 50,
            model_names=args.model_names,  # None → full default
        )
    return defaults


def load_data(tickers, *, synthetic: bool, seed: int) -> dict[str, pd.DataFrame]:
    if synthetic:
        return {t: synthetic_ohlcv(t, n=600, seed=seed) for t in tickers}
    # Live yfinance — only when not in synthetic mode
    from src.data import download_multiple_stocks
    return download_multiple_stocks(tickers)


def main() -> int:
    args = parse_args()
    settings = resolve_settings(args)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" Nested Purged K-Fold runner")
    print("=" * 72)
    print(f" tickers     : {settings['tickers']}")
    print(f" window      : {args.window}")
    print(f" label_mode  : {args.label_mode}")
    print(f" n_outer     : {settings['n_outer']}")
    print(f" n_inner     : {settings['n_inner']}")
    print(f" n_trials    : {settings['n_trials']}")
    print(f" model_names : {settings['model_names']}")
    print(f" synthetic   : {args.synthetic}")
    print(f" use_gpu cfg : {CONFIG.get('use_gpu')}")
    print()

    data = load_data(
        settings["tickers"], synthetic=args.synthetic, seed=args.seed,
    )

    all_oof, all_fold, all_pooled, all_features, all_params = [], [], [], [], []
    skipped: list[str] = []

    for ticker in settings["tickers"]:
        if ticker not in data or data[ticker] is None or len(data[ticker]) == 0:
            print(f"[skip] no data for {ticker}")
            skipped.append(ticker)
            continue
        try:
            result = run_nested_purged_cv(
                data[ticker],
                ticker=ticker,
                window=args.window,
                n_outer_splits=settings["n_outer"],
                n_inner_splits=settings["n_inner"],
                n_trials=settings["n_trials"],
                model_names=settings["model_names"],
                label_mode=args.label_mode,
                seed=args.seed,
                verbose=not args.quiet,
            )
        except Exception as exc:
            print(f"[error] {ticker}: {exc!r}")
            skipped.append(ticker)
            continue

        all_oof.append(result.oof_predictions)
        all_fold.append(result.per_fold_metrics)
        all_pooled.append(result.pooled_metrics)
        all_features.append(result.selected_features)
        all_params.append(result.best_params)

    if not all_oof:
        print("\nNo successful runs. Nothing to write.")
        return 1

    oof_df = pd.concat(all_oof, ignore_index=True)
    fold_df = pd.concat(all_fold, ignore_index=True)
    pooled_df = pd.concat(all_pooled, ignore_index=True)
    feat_df = pd.concat(all_features, ignore_index=True)
    params_df = pd.concat(all_params, ignore_index=True)

    # Combined metrics CSV: per-fold rows tagged "per_fold", pooled rows
    # tagged "pooled" with outer_fold="ALL".
    metrics_combined = pd.concat([
        fold_df.assign(scope="per_fold"),
        pooled_df.assign(outer_fold="ALL", scope="pooled"),
    ], ignore_index=True, sort=False)

    paths = {
        "predictions": results_dir / "nested_cv_predictions.csv",
        "metrics": results_dir / "nested_cv_metrics.csv",
        "features": results_dir / "nested_cv_selected_features.csv",
        "params": results_dir / "nested_cv_best_params.csv",
        "snapshot": results_dir / "nested_cv_run_snapshot.json",
    }

    oof_df.to_csv(paths["predictions"], index=False)
    metrics_combined.to_csv(paths["metrics"], index=False)
    feat_df.to_csv(paths["features"], index=False)
    params_df.to_csv(paths["params"], index=False)

    snap = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tickers_requested": settings["tickers"],
        "tickers_skipped": skipped,
        "window": args.window,
        "label_mode": args.label_mode,
        "n_outer_splits": settings["n_outer"],
        "n_inner_splits": settings["n_inner"],
        "n_trials": settings["n_trials"],
        "model_names": settings["model_names"],
        "seed": args.seed,
        "synthetic": args.synthetic,
        "smoke": args.smoke,
        "use_gpu": CONFIG.get("use_gpu", False),
        "prefer_gpu": CONFIG.get("prefer_gpu", True),
    }
    paths["snapshot"].write_text(json.dumps(snap, indent=2, default=str))

    print()
    print("Outputs:")
    for name, path in paths.items():
        print(f"  {name:>11}: {path}")
    print()
    if not pooled_df.empty:
        print("Pooled OOF accuracy by (ticker, model):")
        for _, row in pooled_df.iterrows():
            print(f"  {row['ticker']:>6}  h={row['window']:>2}  "
                  f"{row['model']:<18}  acc={row['accuracy']:.3f}  "
                  f"AUC={row['auc'] if pd.notna(row['auc']) else float('nan'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
