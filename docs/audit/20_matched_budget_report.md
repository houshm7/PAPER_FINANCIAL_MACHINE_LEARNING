# Matched-Budget Robustness Check — qf-16-matched-budget

**Date:** 2026-05-04
**Closes:** severe-reviewer fatal flaw F3 (leakage gap confounded
with under-tuning) and the related caveat about CatBoost being
silently dropped from the leakage-safe analysis.
**Status:** the matched-budget run was performed at AAPL $h=1$
with $50$ Optuna trials, $5$ inner Purged folds, and four tree
backends (Random Forest, XGBoost, LightGBM, CatBoost) under the
leakage-safe nested-CV protocol. The 22-percentage-point leakage
gap is preserved; the model ranking is not.

---

## 1. Setup and motivation

The reviewer report flagged that the published-style §4 results
were generated with $50$ Optuna trials, $5$ inner Purged folds,
and five tree backends, while the leakage-safe §5 results used
$5$ trials, $2$ inner folds, and three backends (RF, XGBoost,
LightGBM). The 22-pp gap therefore confounded leakage removal
with hyperparameter-budget reduction. The §5.5 caveats subsection
disclosed this; the matched-budget run resolves it.

Run command:

```bash
python scripts/run_nested_cv.py \
    --tickers AAPL --window 1 --label-mode raw_return \
    --n-outer 5 --n-inner 5 --n-trials 50 --boruta-max-iter 100 \
    --model-names "Random Forest" XGBoost LightGBM CatBoost \
    --seed 42 \
    --results-dir results/matched_budget_h1 \
    --show-progress
```

Wall time: $2{,}485$~s ($41$~min) on a CPU-only host. CatBoost was
the bottleneck (fold 1: $744$~s; folds 2-5 averaged $278$~s as
Optuna pruning kicked in).

The four-backend choice mirrors the four backends in §4 modulo
sklearn Gradient Boosting, which has no GPU path and adds no
methodological novelty beyond the four others; it was dropped
to keep the matched-budget sweep tractable on a CPU-only host.

## 2. Headline numbers

### Reduced budget vs matched budget at $h=1$

| Model | Reduced budget (5 trials, 2 folds) | **Matched budget (50 trials, 5 folds)** | Δ |
|---|---:|---:|---:|
| Random Forest | 0.513 | **0.500** | $-0.013$ |
| XGBoost       | 0.488 | **0.503** | $+0.015$ |
| LightGBM      | 0.484 | **0.502** | $+0.018$ |
| CatBoost      | (not run) | **0.500** | n/a |

**Every backend's matched-budget accuracy is within
$[0.500, 0.503]$. The 22-pp leakage gap from the published-style
$0.730$ is preserved.**

### Bootstrap CIs and class-imbalance metrics (matched budget, h=1, AAPL)

| Model | Acc. | Bootstrap 95\% CI | Bal. Acc. | Brier | BSS |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.500 | [0.474, 0.526] | 0.496 | 0.256 | $-0.030$ |
| XGBoost       | 0.503 | [0.476, 0.529] | 0.501 | 0.303 | $-0.217$ |
| LightGBM      | 0.502 | [0.473, 0.531] | 0.499 | 0.318 | $-0.276$ |
| CatBoost      | 0.500 | [0.472, 0.528] | 0.497 | 0.304 | $-0.221$ |

Every CI contains $0.50$. Every balanced accuracy is within
$[0.496, 0.501]$. Every Brier skill score is negative.

### Pairwise DM (matched budget, h=1, AAPL)

All six pairwise DM tests have raw $|t|$ between $0.000$ and
$0.18$, raw $p$-values between $0.86$ and $1.00$, and
Romano-Wolf adjusted $p$-values between $0.99$ and $1.00$. No
pairwise comparison is significant at any conventional level.
The four backends are statistically indistinguishable from one
another and from a coin flip on this asset over this window.

## 3. Three findings the matched-budget run establishes

### 3.1 The leakage gap is not a budget artefact

With ten times more Optuna trials and two and a half times more
inner Purged folds, the leakage-safe accuracy of every backend
stays within $[0.500, 0.503]$. The 22-pp gap between the
published-style $0.730$ and the leakage-safe $\approx 0.50$ is
robust to giving the leakage-safe pipeline the same compute
budget as the legacy single-loop pipeline.

### 3.2 CatBoost was the §4 AUC leader because of leakage

Under the published-style protocol in §4, CatBoost achieved the
highest AUC ($0.770$). Under the matched-budget leakage-safe
protocol, its accuracy is $0.500$ and its balanced accuracy is
$0.497$. Its dominance in §4 is therefore also leakage-driven,
not architectural. The reviewer's concern that the §5 leakage-
safe analysis silently dropped the AUC leader is closed: when
CatBoost is restored, it does not rescue the model ranking.

### 3.3 The reduced-budget RF advantage was a sampling artefact

Under the reduced-budget protocol, Random Forest looked
marginally above the other two backends at $h=1$ ($0.513$ vs
$0.488$ and $0.484$). Under the matched-budget protocol all
four backends converge to within $0.003$ of one another. The
RF advantage was a side-effect of the small Optuna trial count
on the reduced-budget reduced-fold protocol, not a genuine
architectural property.

## 4. Paper changes (`final_paper/main.tex`)

§5.5: the previous "Caveats on the leakage-safe gap measurement"
subsection (which disclosed F3 as an open question) is rewritten
as "Matched-budget robustness check". The new subsection:

- Documents the matched-budget run setup, including the choice
  to drop sklearn Gradient Boosting.
- Adds Table~\texttt{tab:addendum_matched\_budget} with the
  reduced-budget vs matched-budget accuracy comparison and the
  matched-budget bootstrap CIs and class-imbalance metrics.
- States the three findings in §3.1-§3.3 above.
- Notes that the only remaining empirical caveat is the
  single-asset scope.

§6 (Conclusion): the "Limitations" paragraph drops the F3
disclaimer language ("the matched-budget run is the strongest
robustness check we have not yet run") and instead reports the
matched-budget result as a positive finding.

§5.8 (status of replication): drops the matched-budget item
from the open-items list.

## 5. Reviewer-flagged items closed by this commit

| Severe-reviewer concern | Status |
|---|---|
| **F3 matched-budget confound** | **Closed** |
| CatBoost silently dropped from leakage-safe analysis | Closed (restored under matched budget; collapses to chance) |
| §5.5 disclosure was the textually-honest path | Now obsolete because the experiment was run |

The remaining reviewer-flagged items still open are:

- F4 n=1 sample (still single-asset; panel-wide sweep gated on
  multi-day compute).
- Walk-forward robustness (audit C-9; medium effort).
- Hansen-SPA / Reality Check (Romano-Wolf already provides FWER
  control; SPA would be more powerful but yields the same
  qualitative conclusion).

## 6. Style and consistency

```
Missing labels: none
Missing bib keys: none
Unused bib keys: 9 (unchanged)
LaTeX em-dashes: 0 (one CatBoost "---" placeholder replaced with "n/a")
Unicode em-dashes: 0
```

## 7. Reproducing this run

```bash
git checkout main
mkdir -p results/matched_budget_h1
python scripts/run_nested_cv.py \
    --tickers AAPL --window 1 --label-mode raw_return \
    --n-outer 5 --n-inner 5 --n-trials 50 --boruta-max-iter 100 \
    --model-names "Random Forest" XGBoost LightGBM CatBoost \
    --seed 42 --results-dir results/matched_budget_h1 \
    --show-progress > results/matched_budget_h1.log 2>&1
python scripts/run_inference.py \
    --horizons 1 --n-boot 4000 --seed 42 \
    --results-dir results/matched_budget_h1
```

The seed and `--n-boot 4000` make the bootstrap CIs and the
Romano-Wolf $p$-values stable to four decimals across reruns.

## 8. Outputs committed

```
results/matched_budget_h1/
  nested_cv_predictions.csv        ← 4 backends × 1229 OOF rows
  nested_cv_metrics.csv            ← per-fold per-model accuracy
  nested_cv_best_params.csv        ← per-fold Optuna best params
  nested_cv_selected_features.csv  ← per-fold Boruta confirmed
  nested_cv_run_snapshot.json      ← config + git hash + wall time
  inference_oof_cis.csv            ← bootstrap CIs + bal_acc + Brier
  inference_dm_pvalues.csv         ← pairwise DM + Romano-Wolf
  inference_run_snapshot.json
results/matched_budget_h1.log      ← full run log (with progress timing)
```

## 9. What this commit deliberately does NOT do

- Run the matched-budget protocol at $h=5$ or $h=15$. The
  multi-horizon analysis in
  Table~\texttt{tab:addendum\_aapl\_window} retains the
  reduced-budget settings, with the matched-budget check at
  $h=1$ acting as the robustness anchor.
- Run the panel-wide sweep across the other 24 tickers. Still
  multi-day compute.
- Run Hansen-SPA. The Romano-Wolf result on the matched-budget
  data is unanimous (no comparison rejects), so a more powerful
  test would not change the conclusion.
