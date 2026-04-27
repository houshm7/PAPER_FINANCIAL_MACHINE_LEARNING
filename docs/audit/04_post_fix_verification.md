# Post-Fix Verification Report

**Date:** 2026-04-27
**Branches verified:**

| Branch | Topic | HEAD |
|---|---|---|
| `qf-09-severe-review` | Repository audit (`01_repo_audit.md`) | `82df846` |
| `qf-02-cuda-support` | CUDA backends + diagnostic | `e6af48b` |
| `qf-03-label-fixes` | Label construction fixes (`03_label_leakage_report.md`) | `b8f8013` |

The three branches are **not merged**. Each was verified independently
against the user's 13-item checklist, with cross-branch implications
flagged in §3.

---

## 1. Item-by-item checklist

| # | Question | Status | Evidence |
|---|---|---|---|
| 1 | `docs/audit/01_repo_audit.md` exists and is complete | ✅ (after fix) | 370 lines, 10 sections, 26 findings (C-1…C-26) with severity ratings and per-critical fix files. **Was uncommitted in earlier branches** — fixed in this verification step (commit `82df846` on `qf-09-severe-review`). |
| 2 | CUDA support added safely with CPU fallback | ✅ | `src/gpu.py` wraps every probe in `try/except`; `cuda_available()` returns `False` cleanly when torch lacks CUDA; LightGBM `device_type="gpu"` is gated by a cached probe that trains a 32-row dummy model and falls back on any exception. Diagnostic confirms CPU fallback on this CPU-only host. |
| 3 | `src/config.py` contains GPU configuration options | ✅ | `CONFIG["use_gpu"]` (default `True`), `CONFIG["prefer_gpu"]` (default `True`). Read by both `create_models` and (after this verification's fix) `create_dl_models`. |
| 4 | `src/gpu.py` exists | ✅ | Tracked at `qf-02-cuda-support`. Functions: `cuda_available`, `cuda_device_name`, `get_torch_device`, `get_xgboost_gpu_params`, `get_catboost_gpu_params`, `get_lightgbm_gpu_params`, `lightgbm_gpu_supported`, `gpu_summary`. |
| 5 | PyTorch MLP and LSTM use CUDA when available | ✅ (after fix) | Hardcoded `device="cpu"` removed; `_resolve_device()` reads from `gpu.get_torch_device`. **A bug was uncovered during verification:** `create_dl_models` did not pass `device=` to the wrappers, so `CONFIG["use_gpu"]=False` did not propagate. Fixed in commit `e6af48b` on `qf-02-cuda-support`. |
| 6 | XGBoost and CatBoost can use GPU when available | ✅ | XGBoost receives `tree_method="hist", device="cuda"`. CatBoost receives `task_type="GPU", devices="0", bootstrap_type="Bernoulli"` (the explicit `Bernoulli` keeps the existing `subsample` argument legal under the GPU bootstrap). Verified by simulating CUDA in `tests/test_gpu_fallback.py`. |
| 7 | LightGBM GPU optional and safe | ✅ | `lightgbm_gpu_supported()` runs once, caches `True`/`False`, and never raises into the caller. CPU-only LightGBM wheels (the common case on PyPI) keep the model on `device_type="cpu"`. |
| 8 | sklearn `RandomForest` and `GradientBoosting` still CPU-only | ✅ | `test_random_forest_has_no_gpu_kwargs` and `test_sklearn_gradient_boosting_has_no_gpu_kwargs` assert that `device`, `task_type`, `device_type` are absent from those models' params dict, even with `use_gpu=True`. |
| 9 | `create_target_labels` no longer labels unavailable horizons as DOWN | ✅ | New implementation builds a float64 series initialised to `NaN`; only `valid` rows get `±1`. Pinned by `test_create_target_labels_last_window_is_nan[1,2,5,10,15]`. |
| 10 | Raw-return labels are now the default main benchmark | ✅ | `DEFAULT_LABEL_MODE = "raw_return"`; `prepare_features` and `prepare_features_with_t1` default to it. Pinned by `test_label_mode_default_is_raw_return`. |
| 11 | Wavelet labels are only a robustness option | ✅ | `LABEL_MODES = ["raw_return", "wavelet", "exponential", "savgol", "none"]`; module docstring explicitly flags `wavelet` and `savgol` as **non-causal** and reserved for robustness checks. Legacy `smoothing_method=` kwarg retained as deprecated alias. |
| 12 | `X`, `y`, `t1` aligned after dropping unavailable labels | ✅ | Single-pass `dropna` drops both indicator warm-up rows and trailing label-NaN rows; `t1` is computed from the original `df.index` so the prior boundary-cap bug is gone. Pinned by `test_prepare_features_with_t1_index_alignment[1,2,5,10,15]` and `test_prepare_features_with_t1_no_boundary_collapse`. |
| 13 | Tests exist for CUDA fallback and label alignment | ✅ (after fix) | **Before this verification:** label tests existed (21 cases on `qf-03-label-fixes`); CUDA had only a diagnostic script (`scripts/check_gpu.py`), no formal unit tests. **After this verification:** new `tests/test_gpu_fallback.py` (15 cases) on `qf-02-cuda-support`. Combined coverage: 36 unit tests, all passing. |

### Test runs (all on the originating branch)

```
qf-03-label-fixes  : python -m pytest tests/test_label_construction.py -v
                     21 passed in 1.50s
qf-02-cuda-support : python -m pytest tests/test_gpu_fallback.py -v
                     15 passed in 3.41s
qf-02-cuda-support : python scripts/check_gpu.py
                     XGBoost / CatBoost / LightGBM / MLP all [OK] device=cpu
```

---

## 2. Bugs found and fixed during verification

These were not "new methodology" — they were defects in the work
already shipped on `qf-02-cuda-support` and `qf-09-severe-review`:

### V-1. `create_dl_models` ignored `CONFIG["use_gpu"]` [minor]

Tree-model factories read `config.use_gpu` and gate GPU kwargs on it.
`create_dl_models` did not pass any `device=` kwarg, so the
`SklearnMLPClassifier` / `SklearnLSTMClassifier` constructors used their
default `device=None`, which auto-picks `"cuda"` whenever
`torch.cuda.is_available()` — independently of the operator's config.

Demonstrated under simulated CUDA:

```
use_gpu=True  -> XGB device='cuda', MLP._resolve_device()='cuda'
use_gpu=False -> XGB device='cpu',  MLP._resolve_device()='cuda'   <- bug
```

Fix: `create_dl_models` now resolves `device = get_torch_device(prefer_gpu = use_gpu and prefer_gpu)`
and passes it to both wrappers. The previously-dead `prefer_gpu` config
key is now wired in. Commit `e6af48b` on `qf-02-cuda-support`.

### V-2. `docs/audit/01_repo_audit.md` was never committed [minor / hygiene]

The audit document existed only as an untracked working-tree file,
travelling between branches but not reachable from any commit. Fixed in
commit `82df846` on `qf-09-severe-review`.

### V-3. No formal CUDA-fallback unit tests [minor]

Only a diagnostic script existed. Added `tests/test_gpu_fallback.py`
(15 cases) as part of `e6af48b` on `qf-02-cuda-support`. The tests use
`monkeypatch` to simulate CUDA-on/off and verify each helper, the tree
factory, and the DL factory.

---

## 3. Remaining problems (ranked)

### Critical

- **R-1. The three branches are not merged.** Production-readiness
  requires both the label fix (`qf-03`) and the CUDA work (`qf-02`)
  to coexist on a single branch. Until then, no end-to-end run reflects
  the full state. *Action: open a `qf-merge-staging` branch, merge
  `qf-02-cuda-support` and `qf-03-label-fixes` into it, and run the
  combined test suite.*
- **R-2. Audit criticals C-2, C-3, C-4, C-13, C-19, C-22 remain
  unaddressed.** The label fix only resolves the smoothing-related
  half of C-1 plus C-5 and C-7. The Optuna-not-nested critical (C-2),
  feature-selection-on-full-sample critical (C-3), AAPL-tuned-applied-
  panel-wide critical (C-4), live-yfinance-no-snapshot critical (C-13),
  no-economic-backtest critical (C-19), and weak-statistical-inference
  major (C-22) all still hold. The headline 73% number remains suspect
  until at least C-2 + C-3 are fixed.
- **R-3. `final_paper/main.tex` references the old wavelet defaults.**
  The label fix flipped the default to raw returns. The paper was
  intentionally not edited, so the body now contradicts the code's
  default. Once the notebook is rerun, the paper text must be updated.

### Major

- **R-4. `src/tuning.py` does not use the GPU helpers.** Optuna
  reconstructs models manually inside each trial, on CPU only, even
  when `CONFIG["use_gpu"]=True`. On a CUDA host this means hyperparameter
  search is CPU-bound — the dominant compute cost. This was out of
  scope of the qf-02 brief but is a real production gap. *Action:
  refactor `_rf_objective`, `_xgb_objective`, ... to call
  `get_xgboost_gpu_params(use_gpu=...)` etc.*
- **R-5. `pytest` is not declared in `pyproject.toml`.** Reproducers
  must `pip install pytest` manually. Add a `[dependency-groups]
  test = ["pytest>=8"]` block (uv-style) or
  `[project.optional-dependencies] test = ["pytest>=8"]`.
- **R-6. `prepare_features_basak` is silently affected by the
  `create_target_labels` change.** Its docstring still describes the
  old behaviour. With the NaN-aware label fix, the basak path now
  correctly drops the last `window` rows — a behavioural change for
  any saved Basak baseline accuracy. *Action: update the docstring on
  `qf-03-label-fixes` (or a follow-up PR) and re-derive the basak
  baseline numbers when the notebook is rerun.*
- **R-7. `final_notebook/...ipynb` still embeds outputs from the old
  wavelet pipeline.** Cells need to be re-executed before any
  results-section change.

### Minor

- **R-8. `test_wavelet_labels_DO_use_future_smoothing` is data-dependent.**
  It checks `diff_count > 0` on a specific synthetic seed. A stochastic
  edge case could give `0`. *Action: harden by averaging over multiple
  seeds or using a deterministic adversarial price tail.*
- **R-9. `figures/smoothing_comparaison.png` is a typo.** Pre-existing
  in `qf-09:final_paper/main.tex`; cited verbatim by `\includegraphics`.
- **R-10. `CLAUDE.md` is empty (audit C-17), `main.py` is an 84-byte
  stub.** Either populate or delete; both still untracked across the
  three branches. (Note: `CLAUDE.md` is the user's own scratchpad and
  may be intentional — leaving as-is.)
- **R-11. `pyproject.toml` allows future-incompatible upgrades** (audit
  C-15). `uv.lock` is the source of truth for now.
- **R-12. No data snapshot in repo** (audit C-13). yfinance is a moving
  target; today's panel is not necessarily byte-identical to the
  paper's panel. Critical for QF reproducibility but blocked on a
  separate "freeze the data" PR.

---

## 4. Tests / commands to run

Each command must be run on the indicated branch (the work has not
been merged yet).

### Quick sanity (per branch)

```bash
# Label fixes (qf-03-label-fixes)
git checkout qf-03-label-fixes
python -m pytest tests/test_label_construction.py -v
# Expected: 21 passed

# CUDA / GPU fallback (qf-02-cuda-support)
git checkout qf-02-cuda-support
python -m pytest tests/test_gpu_fallback.py -v
# Expected: 15 passed

# CUDA diagnostic (qf-02-cuda-support)
python scripts/check_gpu.py
# Expected on CPU-only host: every backend [OK] with CPU params
# Expected on CUDA host:     every backend [OK] with GPU params (or
#                            CPU for LightGBM if its build is CPU-only)
```

### Cross-branch integration (recommended next step)

```bash
# Stage a merge of the two production branches and rerun all tests.
git checkout main
git checkout -b qf-merge-staging
git merge --no-ff qf-02-cuda-support
git merge --no-ff qf-03-label-fixes
python -m pytest tests/ -v
# Expected: 36 passed

# Smoke-import end-to-end pipeline on the merged branch
python - <<'PY'
from src.config import CONFIG
from src.models import create_models, create_dl_models
from src.preprocessing import LABEL_MODES, DEFAULT_LABEL_MODE
print("LABEL_MODES =", LABEL_MODES)
print("DEFAULT_LABEL_MODE =", DEFAULT_LABEL_MODE)
print("use_gpu cfg =", CONFIG["use_gpu"])
m = create_models(CONFIG)
dl = create_dl_models(CONFIG, input_dim=6)
print("tree models OK:", list(m))
print("dl models OK:", {k: v._resolve_device() for k, v in dl.items()})
PY
```

### Verification of the bug fix (V-1)

```bash
git checkout qf-02-cuda-support
python -m pytest tests/test_gpu_fallback.py::test_create_dl_models_respects_use_gpu_off -v
# Expected: 1 passed (regression test for the DL/config-respect bug)
```

### What NOT to run yet

- Do not rerun the full notebook (`final_notebook/...ipynb`) — that
  belongs to the experiment-rerun task that is explicitly deferred.
- Do not rerun the panel-wide `run_all_stocks_purged_cv`. It would
  burn ~hours of CPU and the headline numbers depend on the still-open
  C-2 / C-3 / C-4 fixes.
- Do not push any of these branches to a public remote until at least
  R-1 (merge) and R-7 (notebook rerun) are done.

---

## 5. Verdict

The three deliverables (audit, CUDA, labels) are **internally
consistent and verified** within their respective branches. One
verification-time bug (V-1: DL ignored `use_gpu=False`) was found and
fixed. One hygiene issue (V-2: audit doc never committed) was found
and fixed. CUDA-fallback unit tests, previously absent, were added.

The repository is **not yet** in a publishable state — the headline
numbers in `final_paper/main.tex` still rest on three unresolved audit
criticals (C-2 nested CV, C-3 selection-in-folds, C-13 data snapshot)
and one major (C-19 economic backtest). Those are tracked by their
respective branches (`qf-04`, `qf-05`, ...) and remain to be done.

The next mechanical step is **R-1**: merge `qf-02-cuda-support` and
`qf-03-label-fixes` into a staging branch and rerun the combined
36-test suite. Everything else flows from there.
