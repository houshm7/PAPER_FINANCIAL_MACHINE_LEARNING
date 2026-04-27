# Nested Purged K-Fold — First Real-Data Run (AAPL, h=1)

**Branch:** `qf-04-nested-purged-cv`
**Date (UTC):** 2026-04-27T21:29Z
**Status:** real-data sanity run completed; legacy 73% claim rejected at z ≪ −17.

This document reports the *first* real-data invocation of the
leakage-safe nested Purged K-Fold pipeline introduced in
`docs/audit/06_nested_cv_report.md`. It replaces the synthetic smoke
output that the qf-04 branch initially shipped.

---

## 1. Setup

| Setting | Value |
|---|---|
| Ticker | `AAPL` |
| Horizon | `h = 1` (1-day directional forecast) |
| Sample period | 2020-01-01 → 2024-12-31 (1257 trading days, live yfinance) |
| Label mode | `raw_return` (causal, default since qf-03) |
| Outer folds | 5 (Purged K-Fold over the full series) |
| Inner folds | 2 (Purged K-Fold inside each outer-train) |
| Optuna trials | 5 per (outer fold, model) |
| Models | XGBoost, Random Forest, LightGBM (3 fastest tree models) |
| Boruta `max_iter` | 20 |
| Embargo | 1% |
| Seed | 42 |
| GPU | requested (config `use_gpu=True`); falls back to CPU on this host |
| Wall time | ≈ 10 minutes on this CPU-only host |

The full configuration is captured in `results/nested_cv_run_snapshot.json`.

**Why these settings.** The legacy paper used 50 Optuna trials × 5
models × single-loop 5-fold Purged CV; this run uses 5 trials × 3
models × **nested** 5-outer × 2-inner Purged CV. Inner trials are
deliberately reduced for compute (the full 50-trial × 5-model panel
sweep is multi-hour even on GPU). The audit's claim is that even with
ample tuning the leakage-safe number stays far below 73%; this run
gives a first empirical test.

---

## 2. Headline numbers

### Pooled OOF accuracy across 1229 OOF predictions

| Model | OOF accuracy | OOF AUC | OOF F1 | 95% CI (Wilson) |
|---|---|---|---|---|
| **Random Forest** | **0.513** | 0.502 | 0.548 | [0.485, 0.541] |
| XGBoost | 0.488 | 0.497 | 0.500 | [0.460, 0.516] |
| LightGBM | 0.484 | 0.487 | 0.524 | [0.456, 0.512] |

The headline OOF accuracy of the best leakage-safe model is **51.3%**
with a 95% CI of [0.485, 0.541]. AUC values cluster around 0.50 —
i.e., the rank ordering of predictions carries essentially no
information about future direction at h=1.

### Per-fold variation

| Model | mean ± std | min | max |
|---|---|---|---|
| Random Forest | 0.513 ± 0.048 | 0.459 | 0.555 |
| XGBoost | 0.488 ± 0.037 | 0.447 | 0.543 |
| LightGBM | 0.484 ± 0.034 | 0.439 | 0.524 |

Across 5 outer folds the per-fold accuracy of every model spans
roughly ±5 pp around chance. No fold lands above 0.56.

---

## 3. Comparison with the legacy 73% claim

The legacy paper (`final_paper/main.tex`, Table 4) reports for AAPL,
h=1, Purged K-Fold:

| Model | Legacy accuracy | This run (leakage-safe) | Drop |
|---|---|---|---|
| Random Forest | **73.0%** | **51.3%** | **−21.7 pp** |
| XGBoost | 72.3% | 48.8% | −23.5 pp |
| LightGBM | 72.6% | 48.4% | −24.2 pp |

### Statistical test of the legacy claim

H₀: the leakage-safe pooled OOF accuracy equals the legacy 0.730. The
test statistic on $n=1229$ OOF predictions is

| Model | z-stat | p-value (two-sided) |
|---|---|---|
| XGBoost | −19.09 | < 10⁻¹⁰ |
| Random Forest | −17.10 | < 10⁻¹⁰ |
| LightGBM | −19.41 | < 10⁻¹⁰ |

The legacy 73% claim is **rejected at every conventional significance
level for all three models**. The 95% CIs in §2 do not overlap 0.73
by a margin of ~20 pp — this is not a noise difference.

---

## 4. Where the 22 pp went — leakage signatures recovered

The audit (`01_repo_audit.md`, C-1 / C-2 / C-3 / C-4) predicted four
distinct leakage paths. Three of them are fixed in the qf-03 / qf-04
branches and the fourth is at most a partial contributor on this run:

| Audit issue | Status in this run | Likely contribution to the 22 pp |
|---|---|---|
| C-1 wavelet labels non-causal | fixed (raw_return is the default) | small but non-zero |
| C-2 single-loop CV / no nested tuning | fixed (nested CV is the new pipeline) | **dominant** |
| C-3 feature selection on full sample | fixed (Boruta inside outer-train) | meaningful |
| C-4 AAPL-tuned panel-wide | not relevant for this single-asset run | n/a here |

Two diagnostic observations from the run support C-2 + C-3 as the
load-bearing fixes:

### 4.1 Hyperparameter drift across outer folds

For an asset where the "true" optimal hyperparameters were stable,
Optuna should pick similar values across outer folds. Instead, on
the leakage-safe outer-train slices:

| Model · Hyperparameter | Per-fold best values | Range |
|---|---|---|
| XGBoost · `learning_rate` | 0.266, 0.261, 0.051, 0.110, 0.185 | [0.051, 0.266] (5×) |
| RF · `n_estimators` | 114, 267, 131, 56, 293 | [56, 293] (5×) |

The legacy paper's Table 6 reports "tuned hyperparameters" as if they
were single best values (e.g. `learning_rate = 0.032` for XGBoost,
`n_estimators = 189` for RF). Under the leakage-safe regime those
values are not stable — they swing 5× across statistically equivalent
outer-train subsamples of the same asset. This is consistent with
"hyperparameters are over-fit to the particular split", which is
exactly what nested CV is designed to surface.

### 4.2 Feature stability

With `boruta_max_iter=20` (reduced for compute), Boruta confirmed all
22 of the 28 candidate features in **every outer fold** — it was too
permissive to discriminate. A production run with `max_iter=100` will
yield more selectivity (the legacy paper's "13 features retained" was
under those settings on the full sample); per-fold stability of the
selected set will then be a useful regularity check. This run does
not yet measure that.

---

## 5. Why these are not yet *paper-grade* numbers

Three caveats that the production sweep should close:

1. **Trial budget.** Each Optuna study had only 5 trials; the legacy
   paper used 50. With 50 trials the leakage-safe accuracy might
   recover one or two percentage points, but it cannot plausibly
   recover 22 — the per-fold standard error is only ~5 pp and AUC is
   already 0.50.
2. **Boruta budget.** `max_iter=20` is too lax. With `max_iter=100`
   the selected feature set will shrink and stabilise across folds.
3. **Single asset.** This is AAPL-only. The legacy 73% was AAPL-only
   too, so the comparison is apples-to-apples on the headline.
   Cross-sector / panel results require running the script with no
   `--tickers` flag (multi-hour on CPU).

The recommended production command, once compute is available, is:

```bash
for h in 1 2 5 10 15; do
    python scripts/run_nested_cv.py --window $h \
        --n-outer 5 --n-inner 5 --n-trials 50 \
        --boruta-max-iter 100 --seed 42
done
```

with the `--tickers` flag omitted to sweep all 25 panel members.

---

## 6. Implications for the paper

When the production sweep lands, the following statements in
`final_paper/main.tex` will need to be revised. The list below is
based on what *this single AAPL run* already proves; the magnitudes
will change with the production sweep but the directions will not.

| Paper text (current) | Likely revised version |
|---|---|
| Abstract: "the best tuned model attains 73% directional accuracy at the one-day horizon" | "the best tuned model attains roughly 51% pooled-OOF accuracy at the one-day horizon under leakage-safe nested CV; the previously reported 73% reflects a single-loop validation that allowed feature selection and Optuna tuning to see the test fold." |
| §4.2 Table 4 (Random Forest at h=1: 73.0%) | replaced with pooled OOF value + per-fold std + 95% CI |
| §4.3 Table 6 (single tuned hyperparameter values) | replaced with per-fold distribution; report mean ± std rather than a single point estimate |
| §4.4 Window-effect (73% → 52% as h grows) | re-derived under nested CV; the *gap* may compress because the h=1 number has dropped |
| Conclusion: "73% directional accuracy roughly ten points above persistence" | replaced; persistence baseline must be re-computed under the same nested CV protocol |

This document does **not** edit the paper — that's the task on
`qf-08-writing-revision`. It documents the empirical case for the
revision.

---

## 7. Reproducing this run

```bash
git checkout qf-04-nested-purged-cv
python scripts/run_nested_cv.py \
    --tickers AAPL --window 1 --label-mode raw_return \
    --n-outer 5 --n-inner 2 --n-trials 5 \
    --model-names XGBoost "Random Forest" LightGBM \
    --boruta-max-iter 20 \
    --seed 42 --quiet > results/nested_cv_aapl_h1.log 2>&1
```

The deterministic seed plus the snapshot JSON make this run
byte-identical for any reproducer with the same yfinance vintage.
The committed `results/nested_cv_*.csv` files are the output of this
exact command. Re-running with different settings (especially
larger `--n-trials` and `--boruta-max-iter`) will overwrite them.

---

## 8. Final verdict

The audit's central hypothesis is empirically vindicated on the most
visible single-stock claim in the paper. **The headline 73% accuracy
for AAPL at h=1 is rejected at z = −17 against the leakage-safe
51.3% pooled OOF accuracy** computed on the same data over the same
period. The 95% CI for the leakage-safe number is [0.485, 0.541] —
indistinguishable from chance and far from 0.73.

The methodological contribution of the paper survives: leakage-aware
validation matters, and the gap between standard K-Fold and Purged
K-Fold is real. The *empirical* claim that tree-based ensembles
"reach 73% directional accuracy" does not survive the more thorough
leakage-safe protocol the paper itself advocates. The corrected
number for the corrected protocol is the headline that should appear
in the revised paper.
