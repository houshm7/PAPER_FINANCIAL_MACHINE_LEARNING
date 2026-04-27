# Repository Audit — Severe Review

**Branch audited:** `qf-09-severe-review`
**Audit date:** 2026-04-27
**Scope:** full repository (`src/`, `final_notebook/`, `final_paper/`, `results/`, configuration, environment)
**Headline claim under review:** "73% directional accuracy at the 1-day horizon under leakage-free evaluation; standard K-fold inflates accuracy by 11–33 pp."

This audit is intentionally severe. The goal is to surface every defect that would block submission to *Quantitative Finance* (QF) or that would invalidate the headline result. Severity legend:

- **critical** — invalidates a headline number, leaks information, or is grounds for desk-rejection.
- **major** — does not invalidate the headline directly but materially weakens the paper or is likely to be raised by referees.
- **minor** — cleanup / cosmetic / quality-of-life.

Verdict up front: **the repository is not QF-ready in its current state.** The two findings labelled C-1 and C-5 (non-causal label smoothing; tuning + feature selection performed once on the full sample) together call into question the reported 73% / 11–33 pp numbers. They must be fixed before any further empirical work is meaningful.

---

## 1. Leakage risks

### C-1. Wavelet label smoothing is non-causal — labels embed future prices [critical]

`src/preprocessing.py:51-104` implements `wavelet_denoising` via `pywt.wavedec` → soft-threshold → `pywt.waverec` on the **entire** closing price series:

- Decomposition coefficients at the fine scales mix neighbours symmetrically; reconstruction is global.
- The universal threshold `sigma * sqrt(2*log(n))` (line 88) is estimated from `np.median(|coeffs[-1]|)/0.6745` over the **whole** signal, i.e. uses post-2024 data when filtering 2020.
- Therefore `\tilde{P}_t = S(P_t)` depends on `P_{t+1}, \dots, P_T`. The label `y_t = 1{\tilde{P}_{t+h} > \tilde{P}_t}` is **measurable with respect to the full future**, not with respect to information available at `t+h`.

This is a leak that the López-de-Prado purging machinery cannot fix, because purging operates on the **label horizon** not on the **smoothing horizon**. With wavelet labels, every observation carries some information about every other observation. Standard *and* purged K-Fold accuracy are both contaminated. The paper's central comparison `tab:kfold` therefore quantifies one form of leakage relative to a leaky baseline.

The paper (`final_paper/main.tex:121-128`, 269) explicitly motivates wavelet smoothing as superior to exponential smoothing because it "preserves sharp transitions" — but does not acknowledge that exponential smoothing is causal whereas the chosen DWT is not. The "67.5% noise reduction" figure is achieved partly by allowing future leakage.

The Savitzky–Golay alternative (`scipy.signal.savgol_filter`, default `mode='interp'`) is **also non-causal** — it is a centred polynomial fit. Only `exponential_smoothing` (lines 31–48) and `method="none"` are causal.

**Files to change later:**

- `src/preprocessing.py` — replace global `pywt.wavedec/waverec` with a causal alternative (e.g. expanding-window wavelet denoising recomputed at each `t`, or simply drop wavelet smoothing for labels and use raw forward returns); restrict `apply_smoothing` to causal methods when used for label construction; mark `savgol_smoothing` as non-causal and forbid for labels.
- `src/preprocessing.py:194-209` (`create_target_labels`) — add a guard / docstring contract that the input series must be ≤-`t`-measurable.
- `final_paper/main.tex:119-129, 269-276` — rewrite the smoothing section once labels are causal; the 67.5% / 78.9% noise-reduction comparison must be recomputed.

### C-2. Tuning is single-level — no nested CV; the reported 73% is not out-of-sample [critical]

`src/tuning.py:191-233` (`tune_model`) runs Optuna over the **entire** `X, y, t1` passed in. In the notebook (`final_notebook/...ipynb` cells around lines 3061-3180) `tune_all_models` is called once on the full AAPL sample, using `_cv_score` (Purged K-Fold) as the objective. The same `X, y` is then re-evaluated with Purged K-Fold downstream (`run_detailed_single_stock_analysis`, `run_all_stocks_purged_cv`).

This is **not nested cross-validation**. Optuna selects hyperparameters by maximising mean Purged-CV accuracy across the entire dataset, and the paper then reports that same maximum (or a re-evaluation on the same folds) as the model's accuracy. The 50-trial maximum over the same folds is upward-biased by an amount that scales with `n_trials` and the variance across folds.

Cf. `final_paper/main.tex:495-545` (Table 5 & 6): the "Tuned" column is the Optuna best value, treated as a generalization estimate.

**Files to change later:**

- `src/tuning.py:191-233` — accept an outer fold index; only tune on training portion of an outer split.
- `src/analysis.py:75-120, 162-214` — wrap evaluation in an outer Purged-CV (or walk-forward) loop; tune inside each outer training fold; report mean-of-outer-folds accuracy.
- `src/models.py:373-399` — adapt `evaluate_with_purged_cv` to accept a per-fold tuning callback.
- `final_paper/main.tex:495-545, 779-783` — the "Tuned" column and the headline 73% must be reported as nested-CV out-of-fold accuracy.

### C-3. Feature selection (Boruta + correlation filter) runs on the full sample [critical]

`src/feature_selection.py:56-81` and `:123-188` (`run_feature_selection`) operate on whole-sample `X, y`. The notebook calls `run_feature_selection(X_fs, y_fs, ...)` exactly once (cell ~2151), producing `selected_features`, which are then handed to every downstream training and evaluation call (e.g. `prepare_features_with_t1(..., feature_cols=selected_features)` at cells 2322, 2593, 3063, 3180+). Selection sees the labels of every future test fold.

Two compounding issues:

1. **Selection leakage.** Boruta uses target labels to test importance. Performing it once on the entire sample lets the future leak into the chosen feature subset, mildly inflating downstream accuracy.
2. **Cross-asset leakage.** Selection is performed on a single ticker (AAPL via `test_ticker` / `ANALYSIS_TICKER`) and the result is then used for all 25 stocks (`run_all_stocks_purged_cv`). The "13 features retained" is an AAPL-2020-2024-conditioned choice masquerading as a panel-wide pipeline.

**Files to change later:**

- `src/feature_selection.py` — make `run_feature_selection` operate on a training subset only; expose a `fit_transform`-style API or accept fold indices.
- `src/analysis.py:75-214` — re-run feature selection inside each outer fold (or accept a global feature list with the caveat documented).
- `final_notebook/...ipynb` (cells 1996-2151, 2322, 2593, 3063) — move feature-selection calls inside the outer-fold loop.
- `final_paper/main.tex:142-144, 281-301` — restate Section 3.3 as "selection is performed inside each outer fold" once the code is fixed; the "13 features" headline must become a distribution.

### C-4. Hyperparameters tuned on AAPL, applied to all panel members and portfolios [critical]

`src/analysis.py:165-214` (`run_all_stocks_purged_cv`) accepts a `hyperparams` dict; the notebook fills it with the Optuna best params from AAPL only. The same applies to `run_portfolio_analysis` (`src/analysis.py:359-410`). The paper (`final_paper/main.tex:785` lim. iv) admits "tuned on a single representative stock and applied uniformly" as a limitation, but the body of the paper presents panel-wide and portfolio-wide accuracy without flagging that those numbers are produced by AAPL-tuned models.

This is leakage of a different flavour — *cross-asset* over-fitting through hyperparameter transfer. It is fixable either by tuning each ticker separately (expensive, also at risk of nesting bug C-2) or by demoting panel-level results to a robustness check rather than the headline cross-sector evidence.

**Files to change later:**

- `src/analysis.py:127-214, 359-410` — add per-ticker (and per-portfolio) tuning, or take an explicit "tune_per_ticker: bool" argument and report results both ways.
- `final_paper/main.tex:601-647` (cross-sector) and `:649-703` (portfolio) — add a row for "AAPL-tuned" vs "self-tuned" hyperparameters.

### C-5. `t1` is built from positional (row-index) arithmetic, not calendar arithmetic [major]

`src/preprocessing.py:349-372` (`prepare_features_with_t1`) computes
`end_positions = np.minimum(np.arange(len(X.index)) + window, len(X.index) - 1)` then `t1 = X.index[end_positions]`.

This is correct only if `X.index` is a contiguous trading-day sequence with no gaps. After `dropna()` removes the warm-up period, the first `max(period)` rows are gone, so the *first* `t1` value is the timestamp `window` rows later in the cleaned frame. That happens to coincide with `t + window` trading days as long as the cleaned frame has no internal gaps, which is the case for daily OHLCV. So this is functionally OK *today*, but the reliance is implicit and fragile.

More importantly the boundary capping `np.minimum(..., len(X.index)-1)` collapses the last `window` values of `t1` to the same final timestamp. Those observations should arguably be dropped (their labels are constructed from a truncated forward window, sub-`h`), not retained with a degenerate `t1`.

**Files to change later:**

- `src/preprocessing.py:194-209, 349-372` — drop the last `window` rows after label creation rather than padding `t1`; or compute `t1` from a calendar offset on `X.index` and drop observations whose label cannot be fully formed.

### C-6. Indicators with global accumulators leak weakly across folds [minor]

`calculate_obv` and `calculate_ad_line` (`src/indicators.py:104-119`, `:208-221`) use `cumsum`. The level (not the change) of these series at time `t` depends on every prior observation; this is fine for causality but means scaling/normalisation choices later in the pipeline (none, currently) could re-introduce leakage. Bollinger %B and CCI use rolling windows that are causal. No leak today, flag for future.

---

## 2. Label construction

### C-7. Tie-handling is silently negative [minor]

`create_target_labels` (`src/preprocessing.py:208`) maps `price_change > 0` to `+1` else `-1`, so exactly-flat horizons (rare on continuous adjusted closes but possible on illiquid stocks at short horizons) become DOWN. Class imbalance reporting "61/39 UP/DOWN" therefore mildly over-counts DOWN. Document and treat ties symmetrically (e.g. drop them or randomise).

### C-8. Class imbalance is real and the headline metric is accuracy [major]

The label balance is ≈61/39 (paper §3.4, line 314). Sample weighting (`compute_sample_weight("balanced", ...)`) is applied at *training* time, but the *evaluation* metric is raw accuracy — which a majority-class predictor would already hit at ≈61%. The paper does report an AUC of 0.755–0.770 (Table 5) which is more honest, but the abstract, conclusion, tables, and headline all foreground accuracy. For a class-imbalanced binary forecasting paper QF would expect log-loss, balanced accuracy, Brier score, or a profit-aware metric as primary.

**Files to change later:**

- `src/models.py:245-302` — promote `roc_auc_score`, `balanced_accuracy`, `log_loss`, and `brier_score_loss` alongside accuracy.
- `final_paper/main.tex` (abstract line 56, §4.2 line 460-486, conclusion 779-783) — re-anchor headline numbers around AUC and balanced accuracy.

### C-9. Persistence baseline is described in the paper but missing in code [major]

`final_paper/main.tex:472` reports a "Persistence" baseline at 63.2% accuracy. There is no `Persistence` class in `src/models.py`, no entry in `BASELINE_NAMES` (`src/config.py:92`), and the notebook does not appear to compute it (no hits for `persistence` in the source). Either the number is computed inline in a notebook cell I missed, or it was hand-entered. Either way, it is not reproducible from `src/`.

**Files to change later:**

- `src/models.py:123-141` — add a `PersistenceClassifier` (predicts `sign(y_{t-1})`) to `create_baseline_models`.
- `src/config.py:92` — add `"Persistence"` to `BASELINE_NAMES`.

---

## 3. Raw-return vs wavelet-smoothed labels

### C-10. No raw-return ablation reported [critical for revision; major today]

The paper switches the label-smoothing operator from "exponential before indicators" (Basak) to "wavelet after indicators". It compares the resulting noise-reduction figures and class balance, but **does not report accuracy under raw (un-smoothed) labels** anywhere. Given that wavelet smoothing is non-causal (C-1), the raw-label run is the only honest baseline.

`src/preprocessing.py` already supports `smoothing_method="none"` (lines 23-24, 155-157), so the experiment is one config-flag away. The fact that this comparison is not in the paper is itself a flag that the wavelet-vs-raw delta is unfavourable, intolerable, or unreported.

**Files to change later:**

- `src/preprocessing.py:247-314` — wire `smoothing_method="none"` end-to-end through `prepare_features_with_t1` (already supported, just needs the experiment).
- `src/analysis.py:31-69` — add a `run_label_construction_ablation(...)` helper that sweeps `{"none", "exponential", "wavelet", "savgol"}` and tabulates accuracy.
- `final_paper/main.tex:119-129, 269-276` — add a Table comparing accuracies under each label-construction operator. If raw-label accuracy is materially below wavelet accuracy, this is exactly the leakage signature predicted in C-1, and the paper's narrative must change.

---

## 4. Purged K-Fold implementation

### C-11. `PurgedKFold` is correct in spirit but slow and uses range-based purging [major]

`src/validation.py:9-72`:

- Inner loop is O(n) per fold with `if j in test_indices` (linear scan over a list); whole split is O(n²). For 25 stocks × 5 windows × ~1250 obs × 5 folds × ~50 Optuna trials × 5 models, the wall-clock is dominated by this.
- Purge condition (line 61): `if obs_t1 >= min_test_t0 and obs_t0 <= max_test_t1`. This uses **the test set's outer envelope** (`min_test_t0`, `max_test_t1`) rather than per-test-observation overlap. For a contiguous test fold this is equivalent to López de Prado's definition; for non-contiguous tests (e.g. combinatorial purged CV) it would be wrong. Document the limitation.
- `_BaseKFold.__init__` is called with `shuffle=False, random_state=None`; sklearn ≥1.4 may warn. OK for now.
- `pct_embargo=0.01` over ~1250 observations is ~12 days. With overlapping labels of horizon `h ∈ {1,2,5,10,15}`, a 12-day embargo is sufficient at `h≤10` and marginal at `h=15`. The paper hard-codes 1% (line 244). Consider reporting sensitivity.
- Embargo is applied **only after** the test fold (line 64). Per López de Prado the embargo can apply on both sides if features are themselves overlapping windows; documented as one-sided in the paper, fine.

**Files to change later:**

- `src/validation.py:9-72` — replace `if j in test_indices` with a boolean mask; vectorise purge using `t1.values`. Add a docstring note on contiguous-test-fold assumption.
- `src/validation.py` — add `CombinatorialPurgedKFold` (López de Prado, Ch. 12) as a robustness option; QF reviewers will ask.
- `final_paper/main.tex:244` — add an embargo sensitivity row.

### C-12. No walk-forward / expanding-window evaluation [major]

Purged K-Fold trains on data from *both before and after* the test fold (with the overlap purged). For trading viability, the standard QF expectation is **walk-forward** (train on `[1, T_k]`, test on `[T_k, T_k + Δ]`, slide). The repo provides `temporal_train_test_split` (`src/validation.py:82-111`) for *one* split but no rolling/expanding evaluator. The paper's own "Future directions" (line 787) admits this.

**Files to change later:**

- `src/validation.py` — add `WalkForwardSplit` (rolling and expanding variants).
- `src/analysis.py` — add `run_walkforward_eval`.
- `final_paper/main.tex` — add a Section 4.x walk-forward table; if the headline 73% drops materially under walk-forward, the abstract/conclusion must be revised.

---

## 5. Whether feature selection happens inside folds

Covered as **C-3** above (critical). To restate the verdict bluntly: feature selection runs **once, on the whole sample, on a single ticker**, before any cross-validation. Every downstream Purged-CV result is conditioned on labels that influenced the chosen feature set. This must be moved inside the outer fold before any "leakage-free" claim survives review.

---

## 6. Whether Optuna tuning is nested

Covered as **C-2** above (critical). To restate: tuning is **not** nested. Optuna optimises mean Purged-CV accuracy on the full sample, and the same sample is then re-evaluated. The "Tuned" column in Table 5 is the Optuna `best_value`, which is upward-biased by `n_trials = 50` selection over the same folds.

The fix is nested CV: outer Purged K-Fold (or walk-forward) for evaluation; inner Purged K-Fold for tuning, restricted to each outer training fold.

---

## 7. Reproducibility

### C-13. Live yfinance dependence; no data snapshot committed [critical]

`src/data.py:59-95` calls `yf.Ticker(...).history(...)`. Yahoo Finance retroactively adjusts splits and dividends and occasionally backfills/corrects history. There is no cached parquet / CSV in the repo. The 25-stock × 5-year panel cannot be reproduced byte-identically by a third party even today, let alone after the paper is published. `auto_adjust=False` is set (good) but does not freeze the upstream data.

**Files to change later:**

- Add a `data/` directory (currently absent) with a frozen panel snapshot (Parquet or CSV.gz), and a `download_or_load` shim in `src/data.py:59-135`.
- Update `.gitignore` (`C:/.../.gitignore`) — currently no data-dir rule.
- `src/data.py` — record the yfinance version, download timestamp, and a checksum next to the snapshot.
- `README.md:64-82` — point reproducers at the snapshot, not at live `jupyter lab`.

### C-14. Determinism gaps [major]

- `src/deep_learning.py:81-101, 138-159` — `torch.manual_seed` is set, but `torch.use_deterministic_algorithms`, cuDNN flags, and the Python / NumPy global seeds are not. MLP/LSTM results will jitter run-to-run.
- `src/feature_selection.py:56-81` — Boruta uses `random_state=42` but its internal RF gets a separate seed; reproducible today, sensitive to library upgrades.
- `numpy.random` is never seeded globally.
- `random.seed` is never set.
- The notebook is not numbered/dated; cell execution order is implicit.

**Files to change later:**

- Add `src/_seeding.py` with a `set_global_seed(seed)` that seeds Python `random`, NumPy, PyTorch (CPU + CUDA), and sets cuDNN deterministic flags.
- Call `set_global_seed(CONFIG["random_state"])` at the top of every entry point in `src/analysis.py` and at the top of the notebook.

### C-15. `pyproject.toml` allows future-incompatible upgrades [minor]

`pyproject.toml` pins lower bounds only (e.g. `numpy>=2.0,<2.4`, `scikit-learn>=1.8.0`, `pandas>=3.0.0`). `uv.lock` is committed (good). Reproducers using `pip install` will get newer packages. Either commit only `uv.lock` workflow in README or add an explicit `requirements-frozen.txt`.

### C-16. Untracked `src/gpu.py` indicates uncommitted work [minor]

`git status` reports `src/gpu.py` as untracked. The file (88 lines) defines GPU helper functions that are **never imported** anywhere else in the repo. Either commit it with a usage path or delete it; do not leave half-finished modules in the working tree.

### C-17. `CLAUDE.md` is empty; `main.py` is a stub [minor]

`CLAUDE.md` is a zero-byte file. `main.py` is an unused 84-byte stub. Either populate or delete.

---

## 8. CUDA usage

### C-18. GPU helpers exist but are dead code; DL hardcoded to CPU [major]

- `src/gpu.py` (untracked) defines `get_xgboost_gpu_params`, `get_catboost_gpu_params`, `get_lightgbm_gpu_params`, `get_torch_device`. None of them is imported by `src/models.py`, `src/tuning.py`, `src/deep_learning.py`, or `src/analysis.py`.
- `src/deep_learning.py:96, 153` — `device="cpu"` is **hardcoded** in both `SklearnMLPClassifier` and `SklearnLSTMClassifier`. Even on a CUDA host, MLP and LSTM run on CPU.
- `src/models.py:75-120` — XGBoost/LightGBM/CatBoost are constructed without `tree_method="hist", device="cuda"` / `task_type="GPU"` / `device_type="gpu"`.

This is not a correctness issue but it is a transparency one: the abstract / methods do not claim GPU, yet `src/gpu.py` exists. Decide explicitly: either remove `gpu.py` and document "CPU-only", or wire it through and document the expected speed-up.

**Files to change later (if going GPU):**

- `src/gpu.py` — commit, add tests.
- `src/models.py:75-120` — call `get_xgboost_gpu_params()`, `get_lightgbm_gpu_params()`, `get_catboost_gpu_params()` at construction, with a `use_gpu: bool` config flag.
- `src/deep_learning.py:96, 153` — replace `device="cpu"` with `device=get_torch_device(prefer_gpu=True)`.
- `src/config.py:26-59` — add `"use_gpu": False` default.
- `README.md:42-62` — note GPU dependency requirements (CUDA toolkit, NVIDIA driver, CUDA-enabled torch wheel).

---

## 9. Economic backtesting gaps

### C-19. No realised P&L, Sharpe, or transaction-cost backtest [critical]

The paper measures classification accuracy. It does **not**:

- Translate predictions into a position (long/short, sizing, risk budget).
- Compute realised P&L, Sharpe, Sortino, Calmar, hit ratio, average win/loss, turnover.
- Apply transaction costs, slippage, or borrow costs.
- Compare against buy-and-hold or a 1/N benchmark on the same universe.
- Decompose returns by regime (COVID crash, 2022 selloff, AI rally).

`src/analysis.py:310-410` (`create_sector_portfolio`, `run_portfolio_analysis`) builds equal-weighted sector portfolios but uses them as **prediction targets**, not as a strategy. `portfolio_stats_df` (Sharpe, MDD) is *passive* portfolio performance, not strategy performance.

The paper's own limitation (line 785, lim. ii) is honest: "directional accuracy does not map directly to net trading profitability". For a methods paper this might be acceptable; for QF it is not. The journal's audience reads "directional forecasting" as a claim about a tradable signal, and asks: net of costs, with what turnover, at what Sharpe, vs what benchmark?

**Files to change later:**

- New module `src/backtest.py` — accepts predicted probabilities + threshold, builds a strategy (e.g., long when `p > 0.5+δ`), computes realised returns net of costs.
- `src/backtest.py` — add metrics: annualised return, vol, Sharpe, max drawdown, hit ratio, turnover, average holding period.
- `src/backtest.py` — add a regime decomposition (sub-period Sharpe, GFC-style) and a buy-and-hold benchmark.
- `final_paper/main.tex` — add a §4.x "Economic significance" with the realised numbers; this is the most-likely-to-be-asked referee question.

### C-20. Class imbalance + balanced sample weights inflate accuracy without economic meaning [major]

Balanced sample weights (`src/models.py:316, 338`, `src/tuning.py:175-176`) up-weight the minority (DOWN) class during training. This *reduces* the trivial majority-classifier ceiling for accuracy — but it also pushes the threshold for predicting UP higher, which for an asymmetric P&L (long-only equity) can hurt realised returns even as accuracy rises. Without a backtest the trade-off is invisible.

---

## 10. Quantitative Finance journal readiness

### C-21. Sample selection issues [major]

- **Survivorship/selection bias.** Twenty-five large-cap survivors of 2024, viewed in 2026, are not a random panel. NVDA, META, TSLA in particular are *ex post* known winners.
- **Sample period is too narrow and too unusual.** 2020-01-01 to 2024-12-31 covers COVID crash, zero-rate, AI rally — five years of structurally unusual regimes. QF reviewers will ask for ≥10 years and out-of-period validation (e.g., reserve 2024 entirely).
- **No fundamental controls.** Pure technical-indicator predictors with no Fama-French-Carhart, sector, or macro overlay. Acceptable as a methodological paper, but the framing must say so.

### C-22. Statistical inference is not rigorous [major]

`src/analysis.py:281-303, 448-481`:

- **ANOVA on accuracy across windows** treats fold-level accuracies as i.i.d. They are not — same stock, overlapping label intervals, common features. F-tests will be anti-conservative.
- **Tukey HSD across model accuracies** likewise assumes independence.
- **No Diebold–Mariano** for pairwise forecast comparison.
- **No Reality Check / Hansen SPA / Romano–Wolf** for multiple-model testing on a common dataset.
- **No White's reality check** vs the persistence baseline.
- **No multiple-comparison correction** for the 5×5 = 25 model-sector cells in §4.6.

Paper claims `"p < 0.0001"` (line 599) for the window effect; with the dependence structure, the effective sample size is much smaller and the *p*-value has no honest interpretation.

**Files to change later:**

- New module `src/inference.py` — implement Diebold–Mariano (with HAC), block-bootstrap CIs for accuracy / Sharpe, Romano–Wolf multi-model adjustment.
- `src/analysis.py:281-303, 448-481` — replace ANOVA / Tukey calls with bootstrap-based tests.
- `final_paper/main.tex:597-599, 638-647` — replace ANOVA *p*-values with DM-test and bootstrap CIs.

### C-23. Missing benchmarks the QF audience expects [major]

- **No comparison vs an autoregressive return model** (AR(1), AR(p), GARCH-mean).
- **No comparison vs a HAR-RV style model** for direction.
- **No "naïve momentum" benchmark** (sign of trailing 1-month return).
- The persistence baseline at 1 day (paper line 472) is not extended to longer horizons.

### C-24. Headline contains numbers that depend on a single asset and a single tuning run [major]

The abstract (`final_paper/main.tex:56`) claims "73% directional accuracy at the 1-day horizon". That number is AAPL-only, AAPL-tuned, single Optuna seed, no nested CV (C-2), label-smoothed by a non-causal filter (C-1). It is also not accompanied by a confidence interval or a *p*-value vs persistence (C-22). For QF the abstract claim must be (a) panel-mean ± CI, (b) net-of-costs, (c) under nested CV with causal labels.

### C-25. Repository hygiene for a published paper [minor]

- `pyproject.toml` name is `"projet"` and description is `"Add your description here"` — auto-generated boilerplate.
- `final_notebook/` contains a single 6 MB `.ipynb` with embedded outputs and absolute Windows paths in the saved cell outputs (e.g. line 4439 `c:\\Users\\houss\\OneDrive - ...`). Strip outputs before publishing or convert to a Quarto / `.py` script.
- `figures/smoothing_comparaison.png` — typo in filename; will be cited verbatim by `\includegraphics`.
- `results/` contains only `purged_cv_results.csv`. No `tuning_results.json`, no fold-by-fold predictions, no SHAP arrays. The audit trail is incomplete; a reproducer cannot verify Table 5 without re-running the notebook.
- `academic_report/`, `journal_qf_template/`, `reference_paper/` are not described in the README.
- README sets a Mac/Linux activation path that uses `source .venv/bin/activate` — but the repo lives on a Windows/OneDrive path; reproducibility on macOS is therefore untested.

### C-26. The paper's own limitations cover roughly half of the issues above [info]

`final_paper/main.tex:785` lists five limitations: (i) feature scope, (ii) transaction costs, (iii) regime, (iv) AAPL-only tuning, (v) wavelet not optimised. The audit's verdict: this list is accurate as far as it goes but is **silent on**: non-causality of the wavelet smoothing (C-1), non-nesting of tuning (C-2), feature-selection leakage (C-3), absence of nested CV / walk-forward (C-12), absence of realised P&L (C-19), and statistical-inference gaps (C-22). These are what referees will demand fixed.

---

## Summary table

| ID    | Severity | Theme                          | Headline                                                                 |
|-------|----------|--------------------------------|--------------------------------------------------------------------------|
| C-1   | critical | leakage / labels               | Wavelet smoothing is non-causal — labels embed full-sample future        |
| C-2   | critical | tuning                         | Optuna tuning is single-level, not nested                                 |
| C-3   | critical | feature selection              | Boruta runs once on the full sample, on a single ticker                   |
| C-4   | critical | tuning transfer                | AAPL-tuned hyperparameters applied panel-wide                            |
| C-5   | major    | t1 construction                | Positional `t1` and boundary capping; brittle, retains degenerate rows    |
| C-6   | minor    | indicators                     | OBV / AD-line cumulative; document causality                              |
| C-7   | minor    | label ties                     | Flat horizons silently classed as DOWN                                    |
| C-8   | major    | metric                         | Headline metric is accuracy under 61/39 imbalance                         |
| C-9   | major    | baseline                       | Persistence baseline reported in paper, missing from `src/`               |
| C-10  | critical | label ablation                 | Raw-return label baseline never reported                                  |
| C-11  | major    | Purged K-Fold                  | O(n²) loop; range-based purging; OK today but document & vectorise        |
| C-12  | major    | validation                     | No walk-forward / expanding-window evaluation                             |
| C-13  | critical | reproducibility / data         | Live yfinance dependence; no committed snapshot                           |
| C-14  | major    | reproducibility / determinism  | Global seeds and cuDNN flags not set                                      |
| C-15  | minor    | reproducibility / deps         | `pyproject.toml` allows future upgrades                                   |
| C-16  | minor    | repo hygiene                   | Untracked `src/gpu.py` is dead code                                       |
| C-17  | minor    | repo hygiene                   | `CLAUDE.md` empty, `main.py` stub                                         |
| C-18  | major    | CUDA                           | GPU helpers exist but unused; DL hardcoded to CPU                         |
| C-19  | critical | economic backtest              | No P&L, no Sharpe, no transaction costs, no regime decomposition          |
| C-20  | major    | metric / weights               | Balanced sample weights inflate accuracy without economic meaning         |
| C-21  | major    | sample                         | Survivorship/selection bias; period too narrow                            |
| C-22  | major    | inference                      | ANOVA / Tukey on dependent folds; no DM, no SPA, no bootstrap             |
| C-23  | major    | benchmarks                     | No AR/GARCH/naïve-momentum benchmarks                                      |
| C-24  | major    | headline                       | 73% is AAPL-only, single seed, no CI, no costs                            |
| C-25  | minor    | repo hygiene                   | Boilerplate `pyproject.toml`, absolute paths in notebook, typo filenames  |
| C-26  | info     | limitations                    | Paper's "Limitations" list misses C-1/2/3/12/19/22                        |

## Recommended fix order before any further experiments

1. C-1 (causal labels) — without this, every accuracy number in the paper is suspect.
2. C-13 (data snapshot) — freeze the panel before fixing anything else, so before/after comparisons are meaningful.
3. C-3 + C-2 + C-4 (selection / tuning leakage) — collapse all three into a single nested-CV refactor inside one outer fold.
4. C-10 (raw-return ablation) — once 1 is fixed, this is one config run.
5. C-19 (economic backtest) — design the strategy, decide costs, run on the now-clean predictions.
6. C-22 (inference) — Diebold–Mariano + block bootstrap once the predictions are stable.
7. Everything else.

Items 1–3 alone are likely to move the headline 73% materially. Plan accordingly.
