# Nested Purged K-Fold Pipeline — Design and Verification

**Branch:** `qf-04-nested-purged-cv`
**Audit cross-references:** C-2 (no nested CV), C-3 (selection outside folds),
C-4 (panel-wide hyperparameters), C-1 (label causality, depends on
`label_mode="raw_return"` from `qf-03`).
**Status:** code shipped, leakage contract enforced by tests, smoke run
produced.

---

## 1. Why this exists

The audit's verification report (`04_post_fix_verification.md`,
remaining problem **R-2**) lists six unresolved criticals. Three of
them — C-2, C-3, C-4 — share a single mechanism: feature selection
and Optuna tuning were performed once on the full sample (or once on
AAPL and applied panel-wide), and the resulting hyperparameters /
feature subset were then re-evaluated on the *same* data via a
single-loop Purged K-Fold. That single-loop design lets the test fold
leak into the choice of hyperparameters and features, which is why
the previously reported 73% headline accuracy at h=1 is not a
genuine out-of-sample number.

The leakage-safe fix is nested cross-validation: the outer Purged
K-Fold is the *evaluator*, the inner Purged K-Fold is the *tuner*,
and feature selection is folded inside the outer training slice so
no test-fold observation ever influences the model that scores it.

This report documents the implementation in `src/pipeline.py` and the
tests that pin the leakage contract.

---

## 2. Algorithm

```
Input:
    df, ticker, h, K_outer, K_inner, n_trials, label_mode,
    seed, model_names

Build (X, y, t1) := prepare_features_with_t1(
    df, window=h, label_mode=label_mode, ...
)
# X.index == y.index == t1.index by construction (audit C-5 fix on qf-03)

For each outer fold k in PurgedKFold(K_outer, t1, embargo):
    train_k_idx, test_k_idx = outer split

    # 1) Slice — outer-test is held back
    X_tr, y_tr, t1_tr = X[train_k_idx], y[train_k_idx], t1[train_k_idx]
    X_te, y_te        = X[test_k_idx],  y[test_k_idx]

    # 2) Feature selection ON OUTER-TRAIN ONLY
    corr_kept   = correlation_filter(X_tr, threshold=0.9)
    confirmed   = boruta_selection(X_tr[corr_kept], y_tr, max_iter=...)
    if not confirmed:                       # graceful fallback
        confirmed = tentative or corr_kept

    X_tr_sel = X_tr[confirmed]
    X_te_sel = X_te[confirmed]              # selection used; no labels

    # 3) Optuna inner tuning ON OUTER-TRAIN ONLY
    For each model_name in model_names:
        best_params[model_name] = tune_model(
            model_name, X_tr_sel, y_tr, t1_tr,
            inner=PurgedKFold(K_inner, t1=t1_tr, embargo)
        )

    # 4) Final fit on outer-train, predict outer-test
    models = create_models(hyperparams=best_params)   # GPU via src.gpu
    For each (name, model):
        fit on (X_tr_sel, y_tr_bin)
        y_pred, y_proba = predict on X_te_sel        # only use of X_te
        record OOF row, per-fold metric, fold's selected features and params

Pool all OOF predictions across folds → pooled metrics (López de Prado's
"all-fold accuracy" — the headline number).
```

**Why this is leakage-safe.** Every label-consuming operation
(`correlation_filter`, `boruta_selection`, `tune_model`) receives a
*positional* slice of the full series whose `.index` is guaranteed
disjoint with `test_k_idx`. The outer Purged K-Fold has already
removed observations whose label interval overlaps with the outer-test
interval (purging) plus an embargo buffer afterwards. The inner
PurgedKFold operates on `t1_tr`, which is the restriction of `t1` to
outer-train indices — so inner purging cannot reach back into
outer-test either. The outer-test slice is read exactly once, by
`predict()`, after the model has already been fitted on outer-train.

**What the new pipeline does *not* solve.** It does not address:

- C-13 (live yfinance, no data snapshot) — orthogonal data-pinning task.
- C-19 (no economic backtest with transaction costs) — separate branch.
- C-22 (statistical inference: ANOVA / Tukey on dependent folds) —
  separate branch.

Those are tracked elsewhere and remain out of scope here.

---

## 3. Files added / changed

| Path | Change |
|---|---|
| `src/pipeline.py` (new) | `run_nested_purged_cv` + `NestedCVResult` dataclass; the canonical research entry point |
| `scripts/run_nested_cv.py` (new) | CLI driver; `--smoke` and `--synthetic` for verification; default = full panel run (multi-hour) |
| `tests/test_nested_pipeline.py` (new) | 6 tests; the four required leakage-contract checks plus an end-to-end smoke |
| `src/analysis.py` | Module-level + per-function legacy banner pointing to `pipeline.run_nested_purged_cv` |
| `results/nested_cv_*.csv`, `nested_cv_run_snapshot.json` (new) | Smoke-run output (synthetic AAPL, 3 outer × 2 inner × 2 trials × 2 models) |

Legacy single-loop helpers in `src/analysis.py`
(`run_kfold_comparison`, `run_detailed_single_stock_analysis`,
`run_single_stock_multiwindow_analysis`, `run_all_stocks_purged_cv`,
`run_portfolio_analysis`) are **kept** per the user's instruction.
Their docstrings now flag them as legacy / non-final and direct
callers to `pipeline.run_nested_purged_cv` for any number that ends
up in the paper.

---

## 4. Defaults

```python
run_nested_purged_cv(
    df, *, ticker, window,
    n_outer_splits=5,         # outer evaluator
    n_inner_splits=5,         # inner tuner
    pct_embargo=0.01,
    n_trials=20,              # Optuna trials per (outer fold, model)
    model_names=TREE_MODEL_NAMES,
    label_mode="raw_return",  # CAUSAL — wavelet only as robustness
    include_changes=True,
    feature_pool="extended",
    corr_threshold=0.9,
    boruta_max_iter=100,
    seed=42,
    verbose=True,
)
```

GPU usage is read from `CONFIG["use_gpu"]` and propagates through the
`src.gpu` helpers via `create_models`, the inner Optuna objectives
(`qf-02:e460ee6`), and the DL device resolution. CPU users see no
behavioural change.

---

## 5. Required outputs (all in `results/`)

| File | Contents |
|---|---|
| `nested_cv_predictions.csv` | OOF predictions: `ticker, outer_fold, date, window, model, y_true, y_pred, y_proba`. One row per (test date, model). |
| `nested_cv_metrics.csv` | Per-fold rows (`scope=per_fold`) with accuracy / precision / recall / specificity / f1 / auc / n_test, plus pooled rows (`scope=pooled, outer_fold=ALL`) computed over the full OOF set. Pooled accuracy is the headline. |
| `nested_cv_selected_features.csv` | Per-fold feature-selection trace: `stage ∈ {corr_dropped, corr_kept, final}` so the reader can audit which features survived correlation filtering and Boruta confirmation in each outer fold. |
| `nested_cv_best_params.csv` | One row per `(ticker, outer_fold, model, param)` showing the Optuna best value. Lets the reader see whether tuned hyperparameters drift across outer folds (a useful regularity check). |
| `nested_cv_run_snapshot.json` | Full configuration of the run: tickers, window, label_mode, fold counts, n_trials, model list, seed, GPU flags, timestamp, smoke/synthetic flags. Required for reproducing or interpreting any committed CSV. |

---

## 6. Tests — leakage contract

`tests/test_nested_pipeline.py` (6 cases, all passing). The four
required tests are:

| # | Test | What it asserts |
|---|---|---|
| 1 | `test_feature_selection_uses_outer_train_only` | For each outer fold k, the index sets passed to `correlation_filter` and `boruta_selection` are **disjoint with that fold's outer-test dates**. (Other folds' test dates are fair game when working on fold k — they are training data in fold k's view.) |
| 2 | `test_optuna_tuning_uses_outer_train_only` | For each outer fold k and model m, the index set passed to `tune_model(m, X, y, t1, ...)` is disjoint with fold k's outer-test dates. The inner PurgedKFold lives entirely inside this slice. |
| 3 | `test_x_y_t1_alignment_after_preparation` + `test_predictions_index_is_subset_of_full_x_index` | `prepare_features_with_t1` returns aligned `(X, y, t1)`; every predicted date is a real date in `X`. (Defence in depth — guards against the audit C-5 boundary-collapse regression.) |
| 4 | `test_outer_test_fold_untouched_until_prediction` | Combined: per-fold disjointness across all label-consuming entry points (corr filter, Boruta, tune_model) **and** every (model, date) appears in OOF predictions exactly once. The test fold IS predicted on; it just is never trained on or tuned on. |

The probes use `monkeypatch.setattr(pipeline_mod, ..., fake)` to
replace the bindings inside `src.pipeline` at the module level so
calls inside `run_nested_purged_cv` go through the recorder. Each
fake records `frozenset(X.index)` and delegates to the real function,
so the test is *both* an interception probe and a real end-to-end
run.

### Run

```
$ python -m pytest tests/test_nested_pipeline.py -v
6 passed in 35.66s

$ python -m pytest                         # full suite, three-branch merged
48 passed in 31.36s   # 17 GPU + 25 label + 6 nested
```

---

## 7. Smoke run

To verify the script end-to-end and produce the four required CSVs
without depending on yfinance, the runner has a `--synthetic` mode
that builds reproducible OHLCV with mild AR(1) momentum:

```
$ python scripts/run_nested_cv.py --synthetic --smoke
 tickers     : ['AAPL']
 window      : 1
 label_mode  : raw_return
 n_outer     : 3
 n_inner     : 2
 n_trials    : 2
 model_names : ['XGBoost', 'Random Forest']
 ...
Pooled OOF accuracy by (ticker, model):
    AAPL  h= 1  XGBoost             acc=0.463  AUC=0.471
    AAPL  h= 1  Random Forest       acc=0.476  AUC=0.453
```

**These accuracy numbers are not the paper headline.** Synthetic
data has no exploitable structure beyond a small AR(1), only 2
Optuna trials and 2 inner folds were used, and there is only one
synthetic ticker. The point of the smoke is to:

1. Verify the script runs end-to-end and writes all five output
   files;
2. Confirm that pooled OOF accuracy on a known-uninformative
   synthetic series stays near chance (50%) — it does (0.46–0.48),
   which is consistent with no leakage;
3. Provide reference output files so the schema is documented.

The committed CSVs in `results/nested_cv_*` are the smoke output and
are clearly tagged in `nested_cv_run_snapshot.json` with
`"synthetic": true, "smoke": true`. **Headline numbers must come
from a non-smoke, non-synthetic run.**

### Counts in the smoke output

| File | Rows | Notes |
|---|---|---|
| `nested_cv_predictions.csv` | 1144 | 3 outer folds × ~191 dates × 2 models |
| `nested_cv_metrics.csv` | 8 | 6 per-fold + 2 pooled |
| `nested_cv_selected_features.csv` | 126 | corr_dropped / corr_kept / final stages × 3 folds |
| `nested_cv_best_params.csv` | 36 | Optuna best params × 3 folds × 2 models |
| `nested_cv_run_snapshot.json` | 1 obj | full config + seed + timestamp |

---

## 8. Running the production pipeline

For paper-grade numbers (multi-hour on CPU; minutes-hours on a
modern GPU once `CONFIG["use_gpu"]=True`):

```bash
# Real-data, single ticker first to sanity-check the live yfinance path:
python scripts/run_nested_cv.py --tickers AAPL --window 1

# Full panel + horizon sweep (run each window separately for clarity):
for h in 1 2 5 10 15; do
    python scripts/run_nested_cv.py --window $h --seed 42
done
```

The CSVs in `results/` will be overwritten each run; copy them aside
or use `--results-dir` for a fresh location if you want to retain a
baseline. The full configuration of every run is captured in the
accompanying `nested_cv_run_snapshot.json`.

---

## 9. What this enables for the paper

Once the production pipeline has run on the real panel, the
following claims become defensible:

- The headline accuracy is the **pooled OOF accuracy** from
  `nested_cv_metrics.csv` rows with `scope="pooled"`. It will almost
  certainly be lower than the previously reported 73% — the bias the
  audit predicted was non-trivial.
- The **per-fold accuracy distribution** (from `scope="per_fold"`
  rows) gives an honest standard error.
- The **feature-set stability across folds** (from
  `nested_cv_selected_features.csv`, `stage="final"`) shows whether
  Boruta picks roughly the same set across outer folds — if it
  doesn't, the previously confident statement that "13 features are
  retained" must be replaced with a distribution.
- The **hyperparameter stability across folds** (from
  `nested_cv_best_params.csv`) is a useful regularity check — if
  `learning_rate` swings by an order of magnitude across outer
  folds, that's a sign of an underspecified search space, not a real
  best value.

These are also the four artefacts a referee will ask for. They are
now produced by a single CLI invocation rather than buried inside a
notebook.
