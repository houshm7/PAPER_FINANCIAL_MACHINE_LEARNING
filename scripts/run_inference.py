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
    balanced_accuracy,
    block_bootstrap_accuracy,
    brier_score,
    brier_skill_score,
    pairwise_diebold_mariano,
    recommended_block_size,
    romano_wolf_dm,
)

RESULTS_DIR = REPO_ROOT / "results"


def _predictions_path_for(horizon: int, base_dir: Path = RESULTS_DIR) -> Path:
    if horizon == 1:
        # Two layouts are supported. The default project layout puts the
        # h=1 predictions directly in the results directory; matched-budget
        # or single-horizon side runs prefer to keep the predictions in
        # the same directory they will write to.
        primary = base_dir / "nested_cv_predictions.csv"
        if primary.exists():
            return primary
        return base_dir / "nested_cv_h1" / "nested_cv_predictions.csv"
    return base_dir / f"nested_cv_h{horizon}" / "nested_cv_predictions.csv"


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
                    if _predictions_path_for(h, base_dir=results_dir).exists()]
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
        path = _predictions_path_for(h, base_dir=results_dir)
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
            # Class-imbalance-aware metrics on the same OOF series.
            # y_proba in the CSV is the probability of UP (label +1).
            yproba_col = sub["y_proba"].astype(float).to_numpy() if "y_proba" in sub.columns else None
            bacc = balanced_accuracy(yt, yp)
            base_rate = float((yt == 1).mean())
            brier = (brier_score(yt, yproba_col, positive_label=1)
                     if yproba_col is not None else float("nan"))
            bss = (brier_skill_score(yt, yproba_col, positive_label=1)
                   if yproba_col is not None else float("nan"))
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
                "balanced_accuracy": bacc,
                "base_rate_pos": base_rate,
                "brier": brier,
                "brier_skill_score": bss,
            })
            print(f"  h={h} {model_name:>14}: acc={ci.point_estimate:.3f} "
                  f"CI=[{ci.lower:.3f}, {ci.upper:.3f}] "
                  f"bacc={bacc:.3f} brier={brier:.3f} bss={bss:+.3f} "
                  f"prior={base_rate:.3f} block={block} n={n}")

        # Pairwise DM across models within each (ticker, horizon),
        # plus Romano-Wolf step-down adjustment of the same family.
        for ticker, sub in preds.groupby("ticker"):
            sub = sub.sort_values(["model", "date"]).reset_index(drop=True)
            losses_by_model: dict[str, np.ndarray] = {}
            for model_name, g in sub.groupby("model"):
                g = g.sort_values("date")
                yt = g["y_true"].astype(int).to_numpy()
                yp = g["y_pred"].astype(int).to_numpy()
                losses_by_model[model_name] = (yt != yp).astype(float)
            # Block size for the per-(model, horizon) bootstrap; we
            # reuse the lag-1 autocorrelation logic from the CI block,
            # taking the maximum block across the participating models
            # so the same resampled time index is consistent for all
            # K = M(M-1)/2 pairs.
            block_sizes = []
            for arr in losses_by_model.values():
                rho_l = (
                    float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
                    if len(arr) > 2 and arr.std() > 0 else 0.0
                )
                block_sizes.append(
                    max(h, recommended_block_size(len(arr), autocorr_lag1=rho_l))
                )
            rw_block = max(block_sizes) if block_sizes else max(h, 5)
            rows = romano_wolf_dm(
                losses_by_model, h=h,
                expected_block_size=rw_block,
                n_boot=args.n_boot, seed=args.seed,
            )
            for row in rows:
                row["ticker"] = ticker
                row["horizon"] = h
                row["rw_block_size"] = rw_block
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

    print("=== Pairwise Diebold-Mariano + Romano-Wolf (negative stat = A has lower loss) ===")
    if not dm_df.empty:
        cols = ["horizon", "model_a", "model_b",
                "dm_stat", "p_value", "rw_p_value",
                "bonferroni_p_value", "n", "lag"]
        cols = [c for c in cols if c in dm_df.columns]
        out = dm_df[cols].copy()
        out["dm_stat"] = out["dm_stat"].round(3)
        for col in ("p_value", "rw_p_value", "bonferroni_p_value"):
            if col in out.columns:
                out[col] = out[col].round(4)
        print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
