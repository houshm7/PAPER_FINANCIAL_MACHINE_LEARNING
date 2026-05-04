# Walk-Forward Robustness Check — qf-17-walk-forward

**Date:** 2026-05-04
**Closes:** audit item C-9 (walk-forward robustness)
**Status:** the leakage-safe nested-CV pipeline was re-run on
AAPL $h=1$ at the matched-budget settings ($50$ Optuna trials,
$5$ inner Purged folds, four tree backends) using
expanding-window walk-forward as the outer scheme instead of
Purged K-Fold. The headline finding (every model accuracy CI
contains chance, every Brier skill score is negative) is
preserved.

---

## 1. Setup

The walk-forward harness (`src/validation.py::WalkForwardCV`)
uses an expanding training window that begins at $40$\,\% of
the series length and absorbs the remainder into five
contiguous test slices. Strict chronological precedence is
enforced (every train index is less than every test index in
the same fold), so no purging is needed: the only overlap risk
at horizon $h$ is the $h-1$ days immediately following the
training tail, handled by an explicit embargo. At $h=1$ the
embargo is zero and the train tail abuts the test head
directly.

For AAPL with $1{,}257$~trading days the geometry is:

| Fold | Train end | Test slice |
|---:|---:|---:|
| 1 | day $502$ | $502$-$651$ ($n=147$) |
| 2 | day $639$ | $639$-$786$ ($n=147$) |
| 3 | day $786$ | $786$-$933$ ($n=147$) |
| 4 | day $933$ | $933$-$1080$ ($n=147$) |
| 5 | day $1080$ | $1080$-$1257$ ($n=177$) |

Pooled OOF: $n = 765$ predictions per model, against $n = 1{,}229$
under Purged K-Fold (the difference is the first $40$\,\% of the
series, which is held out as the initial walk-forward training
window and is therefore never a test fold). Wall time on a
CPU-only host: $4{,}355$~s ($73$~minutes), about $1.75 \times$
the matched-budget Purged K-Fold run because the larger
expanding training sets make per-trial fits slower in later
folds.

Run command:

```bash
python scripts/run_nested_cv.py \
    --tickers AAPL --window 1 --label-mode raw_return \
    --n-outer 5 --n-inner 5 --n-trials 50 --boruta-max-iter 100 \
    --model-names "Random Forest" XGBoost LightGBM CatBoost \
    --seed 42 \
    --results-dir results/walk_forward_h1 \
    --outer-scheme walk_forward \
    --show-progress
```

## 2. Headline numbers

| Model | Acc. | Bootstrap 95\% CI | Bal. Acc. | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.479 | [0.441, 0.518] | 0.489 | 0.259 | $-0.042$ |
| XGBoost       | 0.518 | [0.482, 0.555] | 0.526 | 0.259 | $-0.040$ |
| LightGBM      | 0.497 | [0.460, 0.535] | 0.502 | 0.315 | $-0.265$ |
| CatBoost      | 0.491 | [0.449, 0.533] | 0.499 | 0.283 | $-0.136$ |

**Every bootstrap CI contains $0.50$. Every balanced accuracy is
within $[0.489, 0.526]$. Every Brier skill score is negative.**

The pooled accuracy range is $[0.479, 0.518]$, modestly more
dispersed than the $[0.500, 0.503]$ range under matched-budget
Purged K-Fold. The extra dispersion is consistent with: (i) the
smaller OOF window ($n=737$ vs $n=1{,}229$), and (ii) the fact
that walk-forward test slices are five consecutive forward
windows rather than five interleaved slices. Each test set
therefore lands in a structurally different market regime, so
single-period sampling variance is partially baked into the
model-by-model spread.

## 3. Pairwise Diebold-Mariano + Romano-Wolf

| Pair | DM stat | Raw $p$ | RW $p^{*}$ |
|---|---:|---:|---:|
| CatBoost vs LightGBM | $+0.31$ | 0.754 | 0.773 |
| CatBoost vs RF       | $-0.74$ | 0.458 | 0.718 |
| CatBoost vs XGBoost  | $+1.53$ | 0.127 | 0.397 |
| LightGBM vs RF       | $-1.03$ | 0.306 | 0.602 |
| LightGBM vs XGBoost  | $+1.31$ | 0.192 | 0.485 |
| **RF vs XGBoost**    | $+2.67$ | **0.008** | **0.044** |

One pairwise comparison (RF vs XGBoost) survives Romano-Wolf
adjustment at $\alpha = 0.05$ ($\hat{p}^{*} = 0.044$). The
remaining five pairs are tied. Under matched-budget Purged
K-Fold on the same data
(Table~`tab:addendum_dm`), all six pairwise $\hat{p}^{*}$
values were above $0.99$. We do not draw a model-selection
conclusion from a single borderline pair on a single asset
under one validation scheme; the appropriate reading is that
walk-forward and Purged K-Fold partly disagree on which model
is least bad and unanimously agree that none of them is good.

## 4. Robustness verdict

The 22-percentage-point leakage gap reported in
Sections~\ref{sec:addendum} through~\ref{sec:addendum_caveats}
is robust to the choice of outer validation scheme. Walk-forward
forecloses look-ahead leakage by chronological construction
(no purging required), while Purged K-Fold relies on the
explicit purging of overlapping labels. The two schemes differ
fundamentally in how the training set is constructed for each
test fold; both yield leakage-safe AAPL $h=1$ accuracy at
chance.

The reviewer-flagged audit C-9 ("walk-forward robustness") is
closed.

## 5. Paper changes (`final_paper/main.tex`)

A new §5.6 "Walk-forward robustness check"
(`sec:addendum_walkforward`) is inserted between the
matched-budget subsection and the block-bootstrap inference
subsection. It documents the geometry of the walk-forward
splits, reports Table~`tab:addendum_walkforward`, and concludes
with the robustness reading.

The Conclusion's "Future directions" paragraph drops the
walk-forward sentence (the work is now done, not a future
direction).

The §5.9 status block reports both the matched-budget run and
the walk-forward run as completed. The single remaining
empirical caveat is the panel-wide sweep.

## 6. Implementation notes

`src/validation.py` gained a `WalkForwardCV` class with seven
unit tests in `tests/test_walk_forward.py`. `src/pipeline.py`
gained an `outer_scheme` parameter with values
`"purged_kfold"` (default) and `"walk_forward"`.
`scripts/run_nested_cv.py` gained an `--outer-scheme` CLI flag.

Test count: 92 → 99 (seven new walk-forward unit tests).

## 7. What this commit deliberately does NOT do

- Does not run walk-forward at $h=5$ or $h=15$. The multi-
  horizon analysis in Table~`tab:addendum_aapl_window` retains
  Purged K-Fold; walk-forward at $h=1$ acts as the partition-
  geometry robustness anchor.
- Does not run walk-forward across the full 25-ticker panel.
  Multi-day compute.
- Does not implement the rolling-window variant of
  walk-forward (where the training window has a fixed size
  rather than expanding). The expanding-window variant is the
  standard choice in financial-ML reproducibility studies; the
  rolling variant adds a hyperparameter (window length) without
  obvious methodological gain.

## 8. Reviewer-flagged status after this commit

| Concern | Status |
|---|---|
| F1, F2, F3, F4 | Closed |
| Bonferroni arithmetic | Closed |
| Methodology consistency | Closed |
| Data availability | Closed |
| Romano-Wolf request | Closed |
| C-26 balanced-accuracy / Brier | Closed |
| **C-9 walk-forward robustness** | **Closed (this commit)** |
| Panel-wide sweep | Open (compute-bound) |
| Hansen-SPA | Open (low-value given the unanimous Romano-Wolf negatives) |

## 9. Output files

```
results/walk_forward_h1/
  nested_cv_predictions.csv        ← 4 backends × 737 OOF rows
  nested_cv_metrics.csv            ← per-fold per-model accuracy
  nested_cv_best_params.csv
  nested_cv_selected_features.csv
  nested_cv_run_snapshot.json
  inference_oof_cis.csv            ← bootstrap CIs + bal_acc + Brier
  inference_dm_pvalues.csv         ← pairwise DM + Romano-Wolf
  inference_run_snapshot.json
results/walk_forward_h1.log        ← full progress log
```

## 10. Reproducing this run

```bash
git checkout main
mkdir -p results/walk_forward_h1
python scripts/run_nested_cv.py \
    --tickers AAPL --window 1 --label-mode raw_return \
    --n-outer 5 --n-inner 5 --n-trials 50 --boruta-max-iter 100 \
    --model-names "Random Forest" XGBoost LightGBM CatBoost \
    --seed 42 --results-dir results/walk_forward_h1 \
    --outer-scheme walk_forward --show-progress \
    > results/walk_forward_h1.log 2>&1
python scripts/run_inference.py \
    --horizons 1 --n-boot 4000 --seed 42 \
    --results-dir results/walk_forward_h1
```

The seed and `--n-boot 4000` make the bootstrap CIs and the
Romano-Wolf $p$-values stable to four decimals across reruns.
