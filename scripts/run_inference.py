"""Apply leakage-aware inference to the existing OOF predictions.

Reads the per-horizon ``nested_cv_predictions.csv`` files written by
``scripts/run_nested_cv.py``, runs

  - block-bootstrap CIs on pooled OOF accuracy (replacing the Wilson
    asymptotic CIs in the paper addendum);
  - pairwise Diebold-Mariano tests across the three tree models on
    the per-observation 0/1 loss;

and writes:

  results/inference_oof_cis.csv
  results/inference_dm_pvalues.csv
  results/inference_run_snapshot.json

Usage::

    python scripts/run_inference.py --horizons 1 5 15 --n-boot 4000
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

from src.inference import (  # noqa: E402
    block_bootstrap_accuracy,
    pairwise_diebold_mariano,
    recommended_block_size,
)

RESULTS_DIR = REPO_ROOT / "results"


def _predictions_path_for(horizon: int) -> Path:
    if horizon == 1:
        return RESULTS_DIR / "nested_cv_predictions.csv"
    return RESULTS_DIR / f"nested_cv_h{horizon}" / "nested_cv_predictions.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference (block-bootstrap + DM) runner")
    p.add_argument("--horizons", nargs="*", type=int, default=None,
                   help="Horizons to process (default: all available)")
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Discover horizons from filesystem if not specified.
    horizons = args.horizons
    if horizons is None:
        horizons = [h for h in (1, 2, 5, 10, 15)
                    if _predictions_path_for(h).exists()]
    if not horizons:
        print("No predictions found. Run scripts/run_nested_cv.py first.",
              file=sys.stderr)
        return 1

    print("=" * 72)
    print(" Leakage-aware inference (block bootstrap + Diebold-Mariano)")
    print("=" * 72)
    print(f" horizons : {horizons}")
    print(f" n_boot   : {args.n_boot}")
    print(f" alpha    : {args.alpha}")
    print(f" seed     : {args.seed}")
    print()

    ci_rows: list[dict] = []
    dm_rows: list[dict] = []

    for h in horizons:
        path = _predictions_path_for(h)
        preds = pd.read_csv(path, parse_dates=["date"])
        # Predictions are tagged with model; we treat each (ticker,
        # model) as an independent series.
        for (ticker, model_name), sub in preds.groupby(["ticker", "model"]):
            sub = sub.sort_values("date").reset_index(drop=True)
            yt = sub["y_true"].astype(int).to_numpy()
            yp = sub["y_pred"].astype(int).to_numpy()
            n = len(yt)
            # Block size: at least h-1 + 1 to span the overlap, and
            # enlarged for any persistence in the correctness series.
            corr = (yt == yp).astype(float)
            rho = (
                float(np.corrcoef(corr[:-1], corr[1:])[0, 1])
                if n > 2 and corr.std() > 0 else 0.0
            )
            block = max(h, recommended_block_size(n, autocorr_lag1=rho))

            ci = block_bootstrap_accuracy(
                yt, yp,
                expected_block_size=block,
                n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
            )
            ci_rows.append({
                "ticker": ticker, "horizon": h, "model": model_name,
                "n": n,
                "accuracy": ci.point_estimate,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "alpha": args.alpha,
                "block_size": block,
                "autocorr_lag1": rho,
                "n_boot": args.n_boot,
            })
            print(f"  h={h} {model_name:>14}: acc={ci.point_estimate:.3f} "
                  f"CI=[{ci.lower:.3f}, {ci.upper:.3f}] "
                  f"block={block} n={n}")

        # Pairwise DM across models within each (ticker, horizon).
        for ticker, sub in preds.groupby("ticker"):
            sub = sub.sort_values(["model", "date"]).reset_index(drop=True)
            losses_by_model: dict[str, np.ndarray] = {}
            for model_name, g in sub.groupby("model"):
                g = g.sort_values("date")
                yt = g["y_true"].astype(int).to_numpy()
                yp = g["y_pred"].astype(int).to_numpy()
                losses_by_model[model_name] = (yt != yp).astype(float)
            rows = pairwise_diebold_mariano(losses_by_model, h=h)
            for row in rows:
                row["ticker"] = ticker
                row["horizon"] = h
            dm_rows.extend(rows)

        print()

    cis_df = pd.DataFrame(ci_rows)
    dm_df = pd.DataFrame(dm_rows)

    cis_path = results_dir / "inference_oof_cis.csv"
    dm_path = results_dir / "inference_dm_pvalues.csv"
    snap_path = results_dir / "inference_run_snapshot.json"

    cis_df.to_csv(cis_path, index=False)
    dm_df.to_csv(dm_path, index=False)

    snap = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "horizons": horizons,
        "n_boot": args.n_boot,
        "alpha": args.alpha,
        "seed": args.seed,
        "n_ci_rows": int(len(cis_df)),
        "n_dm_rows": int(len(dm_df)),
    }
    snap_path.write_text(json.dumps(snap, indent=2))

    print(f"Wrote {cis_path}")
    print(f"Wrote {dm_path}")
    print(f"Wrote {snap_path}")
    print()

    print("=== Pairwise Diebold-Mariano (negative stat = A has lower loss) ===")
    if not dm_df.empty:
        out = dm_df[["horizon", "model_a", "model_b",
                     "dm_stat", "p_value", "n", "lag"]].copy()
        out["dm_stat"] = out["dm_stat"].round(3)
        out["p_value"] = out["p_value"].round(4)
        print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
