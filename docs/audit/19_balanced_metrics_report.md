# Class-Imbalance-Aware Metrics — qf-15-balanced-metrics

**Branch:** `qf-15-balanced-metrics`
**Closes:** audit item C-26 (balanced-accuracy / Brier
follow-up) and the §5.4 paper-text hedge that the $h=15$
elevation might be class-imbalance exploitation rather than
directional skill.
**Status:** balanced accuracy, Brier score, and Brier skill
score added to `src/inference.py`, computed for every
(model, horizon) combination on the leakage-safe AAPL OOF
predictions, and folded into a new
Table~`tab:addendum_imbalance_metrics`.

---

## 1. New module additions

`src/inference.py` gains three pure-function metrics:

```python
balanced_accuracy(y_true, y_pred) -> float
brier_score(y_true, y_proba_pos, *, positive_label=1) -> float
brier_skill_score(y_true, y_proba_pos, *, positive_label=1) -> float
```

- Balanced accuracy is the mean of per-class recall, invariant
  to the marginal class prior. A constant majority-class
  predictor scores $0.5$ regardless of imbalance level.
- Brier is the mean squared error of the probability forecast
  against the binary indicator $y_t = \mathbf{1}\{r_{t,h} > 0\}$.
- Brier skill score (BSS) is the relative reduction in Brier
  versus a constant predictor at the empirical base rate.
  Negative BSS means the probability forecast is worse than
  predicting the base rate every period.

## 2. Tests (7 new cases, all passing)

`tests/test_inference.py` count: 18 → 25.

| # | Property |
|---|---|
| 19 | `balanced_accuracy` on perfect prediction is $1.0$. |
| 20 | `balanced_accuracy` on a constant majority-class predictor for an 80/20 imbalance is exactly $0.5$ while raw accuracy is $0.8$. |
| 21 | A 50/50 random predictor on imbalanced data has balanced accuracy near $0.5$ in expectation. |
| 22 | `brier_score` is $0$ when the probability forecast equals the indicator exactly. |
| 23 | `brier_score` of a constant $0.5$ on balanced labels is $0.25$. |
| 24 | `brier_skill_score` of a base-rate predictor on imbalanced data is $0$. |
| 25 | `brier_skill_score` of a forecast better than base rate is positive. |

Full suite: 92/92 pass (was 85, now 85 + 7).

## 3. Application to AAPL OOF predictions

`scripts/run_inference.py` was extended to compute the three
metrics for every (model, horizon) and write them into
`results/inference_oof_cis.csv`. Re-ran with
`--horizons 1 5 15 --n-boot 4000 --seed 42`.

### 3.1 Headline numbers

| Horizon | Model | Acc. | Bal. Acc. | Brier | BSS | Base rate |
|---|---|---:|---:|---:|---:|---:|
| $h=1$  | LightGBM      | 0.484 | 0.469 | 0.291 | $-0.166$ | 0.532 |
| $h=1$  | Random Forest | 0.513 | 0.495 | 0.299 | $-0.197$ | 0.532 |
| $h=1$  | XGBoost       | 0.488 | 0.489 | 0.299 | $-0.200$ | 0.532 |
| $h=5$  | LightGBM      | 0.479 | 0.468 | 0.343 | $-0.400$ | 0.572 |
| $h=5$  | Random Forest | 0.516 | 0.492 | 0.273 | $-0.115$ | 0.572 |
| $h=5$  | XGBoost       | 0.478 | 0.465 | 0.302 | $-0.234$ | 0.572 |
| $h=15$ | LightGBM      | 0.530 | 0.464 | 0.387 | $-0.638$ | 0.616 |
| $h=15$ | Random Forest | 0.523 | 0.462 | 0.282 | $-0.194$ | 0.616 |
| $h=15$ | **XGBoost**   | **0.560** | **0.495** | 0.283 | $-0.195$ | 0.616 |

### 3.2 Headline reading

The §5.4 paper-text hedge that "the elevated accuracy partly
reflects the model successfully exploiting class imbalance via
majority-class prediction rather than a genuine directional
signal" is now a measured fact, not a hedge:

- **At $h=15$, XGBoost balanced accuracy is $0.495$.** The
  56\,\% accuracy is class-imbalance exploitation, not
  directional skill.
- **All nine (model, horizon) combinations have negative Brier
  skill scores.** None of the leakage-safe probability
  forecasts is better-calibrated than a constant prediction at
  the base rate.
- The $h=15$ class prior is $0.616$, materially imbalanced;
  $h=5$ is $0.572$; $h=1$ is $0.532$. Imbalance grows with
  horizon, as expected from longer windows inheriting the
  upward equity drift.

This is a stronger empirical statement than the previous
bootstrap-CI evidence alone: it says not only that accuracy
is statistically tied with chance at $h \ge 5$, but also that
the probability forecasts themselves are useless even after
accounting for the base rate.

## 4. Paper changes

§5.4 (multi-horizon comparison):

- The "should be read carefully" paragraph is rewritten. The
  previous "deferred to the metric-upgrade audit item C-26"
  hedge is replaced with the measured numbers and the explicit
  conclusion that the $h=15$ elevation is class-imbalance
  exploitation.
- New Table~`tab:addendum_imbalance_metrics` lists Acc., Bal.
  Acc., Brier, BSS, and base rate for all nine (model,
  horizon) combinations. Caption is self-contained.
- A new closing paragraph notes that all nine BSS values are
  negative, i.e., the leakage-safe probability forecasts are
  uniformly worse-calibrated than a base-rate predictor.

§6 (Conclusion):

- The "Limitations" paragraph drops the "Balanced-accuracy
  and Brier metrics, ... deferred to a metric-upgrade
  revision" sentence. The metric upgrade is no longer
  outstanding.

§5.8 (status of panel replication):

- The "remaining open audit items are secondary
  (balanced-accuracy and Brier metrics, walk-forward
  robustness, matched-budget run, panel-wide sweep)" sentence
  drops the balanced-accuracy / Brier item. The remaining
  open items are walk-forward robustness, matched-budget
  run, panel-wide sweep.

## 5. Style and consistency

```
Missing labels: none
Missing bib keys: none
Unused bib keys: 9 (unchanged)
LaTeX em-dashes (---): 0
Unicode em-dashes: 0
```

## 6. Audit closure status after this commit

| Item | Status |
|---|---|
| C-1 to C-22 | All closed |
| Reviewer F1, F2, F4, Bonferroni, methodology, data-availability, Romano-Wolf | All closed (qf-12, qf-13, qf-14) |
| Reviewer F3 matched-budget confound | Disclosed (qf-12); not yet run |
| **Audit C-26 (balanced-accuracy / Brier)** | **Closed on this branch** |
| Walk-forward robustness (C-9) | Open |
| Matched-budget run | Open (compute) |
| Panel-wide nested-CV sweep | Open (compute) |

## 7. Reproducing this run

```bash
git checkout qf-15-balanced-metrics
python -m pytest tests/test_inference.py -v
python scripts/run_inference.py --horizons 1 5 15 --n-boot 4000 --seed 42
```

The metrics are computed by deterministic post-processing of
the committed `nested_cv_predictions.csv` files; the only
randomness is in the existing block-bootstrap CIs, which use
seed 42.

## 8. What this branch deliberately does NOT do

- Does not run the matched-budget leakage-safe protocol. Still
  the strongest open robustness check.
- Does not run the panel-wide nested-CV sweep or apply
  balanced-accuracy / Brier across the panel. Single-asset
  evidence only.
- Does not implement reliability diagrams or calibration plots
  (would visualise the negative BSS findings; deferred to the
  figures pass if needed).
- Does not run Hansen-SPA. Romano-Wolf already says nothing
  rejects.
