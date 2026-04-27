# Tuning GPU-Bypass Fix — Report

**Branch:** `qf-02-cuda-support`
**Audit cross-reference:** `docs/audit/04_post_fix_verification.md` issue **R-4**
**Status:** code fixed, tests added, pytest declared.

---

## 1. Problem

Verification (`04_post_fix_verification.md` §3) found that
`src/tuning.py` reconstructed every estimator manually inside the
Optuna objective and never called the helpers in `src/gpu.py`. As a
consequence, on a CUDA host the rest of the pipeline (`create_models`,
`create_dl_models`) ran on GPU but Optuna tuning fell back to CPU
silently. Since hyperparameter search is the dominant compute cost
(50 trials × 5 models × 5 windows × 25 stocks), this defeated most of
the GPU benefit.

The bypass affected three tree backends — XGBoost, LightGBM, CatBoost
— and the two DL backends (MLP, LSTM). Random Forest and sklearn
`GradientBoostingClassifier` have no native CUDA path and were
correctly left on CPU.

---

## 2. Fix

`src/tuning.py` now imports from `src.gpu` and applies the same
backend-specific kwargs as `create_models` / `create_dl_models`:

```python
from .gpu import (
    get_catboost_gpu_params,
    get_lightgbm_gpu_params,
    get_torch_device,
    get_xgboost_gpu_params,
)

def _xgb_objective(trial, X, y, t1, n_splits, pct_embargo, config):
    ...
    model = xgb.XGBClassifier(
        **params, objective="binary:logistic",
        random_state=config["random_state"], n_jobs=config["n_jobs"],
        **get_xgboost_gpu_params(use_gpu=config.get("use_gpu", False)),
    )
    ...

def _lgb_objective(trial, X, y, t1, n_splits, pct_embargo, config):
    ...
    model = lgb.LGBMClassifier(
        **params, is_unbalance=True,
        random_state=config["random_state"],
        n_jobs=config["n_jobs"], verbose=-1,
        **get_lightgbm_gpu_params(use_gpu=config.get("use_gpu", False)),
    )
    ...

def _catboost_objective(trial, X, y, t1, n_splits, pct_embargo, config):
    ...
    model = CatBoostClassifier(
        **params, auto_class_weights="Balanced",
        random_state=config["random_state"],
        verbose=False, thread_count=config["n_jobs"],
        **get_catboost_gpu_params(use_gpu=config.get("use_gpu", False)),
    )
    ...

def _resolve_torch_device(config):
    use_gpu = config.get("use_gpu", False)
    prefer_gpu = config.get("prefer_gpu", True)
    return get_torch_device(prefer_gpu=use_gpu and prefer_gpu)

# _mlp_objective / _lstm_objective pass device=_resolve_torch_device(config)
# to the SklearnMLPClassifier / SklearnLSTMClassifier constructors.
```

### Properties preserved

- **Search spaces unchanged.** Every `trial.suggest_*` call is
  byte-identical to pre-fix; only the model constructor gained a
  `**get_*_gpu_params(...)` (or `device=`) kwarg. No hyperparameter
  ranges were widened, narrowed, or renamed, so historical Optuna
  studies remain comparable.
- **CPU fallback.** Each helper short-circuits to CPU when
  `use_gpu=False` or `cuda_available()` returns False. LightGBM
  additionally probes the installed build with a 1-iteration training
  on a 32-row dummy dataset; CPU-only wheels (the common pip case)
  resolve to `device_type="cpu"` instead of crashing mid-training.
- **CatBoost subsample compatibility.** `get_catboost_gpu_params`
  injects `bootstrap_type="Bernoulli"` on GPU so the existing
  `subsample` argument in the CatBoost search space (`tuning.py:105`)
  remains legal — without this, CatBoost on GPU defaults to
  `bootstrap_type="Bayesian"`, which raises on `subsample`.
- **Random Forest and sklearn Gradient Boosting are untouched.**
  `_rf_objective` (`tuning.py:32-44`) and `_gb_objective`
  (`tuning.py:65-77`) have no GPU helper calls. Sklearn has no native
  CUDA path; injecting GPU kwargs there would either be ignored or
  raise.

---

## 3. Tests

Added to `tests/test_gpu_fallback.py`:

```python
def test_tuning_objectives_apply_gpu_kwargs(fake_cuda, monkeypatch):
    cfg = {**CONFIG, "use_gpu": True}
    xgb_m, lgb_m, cb_m = _run_objectives(monkeypatch, cfg)
    assert getattr(xgb_m, "device", None) == "cuda"
    assert lgb_m.get_params().get("device_type") == "gpu"
    assert cb_m.get_param("task_type") == "GPU"
    assert cb_m.get_param("bootstrap_type") == "Bernoulli"


def test_tuning_objectives_respect_use_gpu_off(fake_cuda, monkeypatch):
    cfg = {**CONFIG, "use_gpu": False}
    xgb_m, lgb_m, cb_m = _run_objectives(monkeypatch, cfg)
    assert getattr(xgb_m, "device", None) == "cpu"
    assert lgb_m.get_params().get("device_type") == "cpu"
    assert cb_m.get_param("task_type") == "CPU"
```

Both tests use `monkeypatch` to simulate a CUDA host *without* needing
real hardware: `fake_cuda` patches `cuda_available()` to return `True`
and short-circuits the LightGBM build probe. A stub `_StubTrial`
returns the midpoint / first option for every `suggest_*` call so the
objective just builds *some* valid model. `_cv_score` is monkeypatched
to capture the constructed estimator and return a constant score —
the actual training loop is never invoked, so the tests run in a few
seconds.

The "real CPU host" smoke test (`test_real_cpu_host_smoke`) covers the
unpatched path: on this CPU-only Windows host, `cuda_available()`
returns False and every helper resolves to CPU kwargs, so the tuning
objectives also fall back transparently.

### Test runs (current `qf-02-cuda-support` HEAD `e460ee6`)

```
$ python -m pytest tests/test_gpu_fallback.py::test_tuning_objectives_apply_gpu_kwargs \
                   tests/test_gpu_fallback.py::test_tuning_objectives_respect_use_gpu_off -v
2 passed in 4.00s

$ python -m pytest                            # whole suite via pyproject.toml testpaths
17 passed in 3.78s

$ python scripts/check_gpu.py                  # diagnostic
... XGBoost / CatBoost / LightGBM / MLP all [OK] device=cpu (no CUDA on this host)
```

Combined with the label suite on the merge-staging branch:

```
$ git checkout qf-merge-staging && python -m pytest
42 passed in 5.97s   # 17 GPU + 25 label
```

---

## 4. pytest declaration

`pyproject.toml` gained:

```toml
[dependency-groups]
test = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

Reproducers can run `uv sync --group test` (uv-style) or
`pip install pytest>=8`. Bare `pytest` now picks up the `tests/`
directory automatically.

---

## 5. What this PR does **not** change

Per the task constraints:

- No full Optuna tuning was run — only the 1-iteration probe trains
  inside `lightgbm_gpu_supported()`.
- The notebook (`final_notebook/...ipynb`) was not re-executed.
- `final_paper/main.tex` was not edited.
- Scientific methodology is unchanged. The only behavioural difference
  visible to a CPU-only user is that bare `pytest` now discovers the
  test suite automatically; no model training path changed.

---

## 6. Open follow-ups

The verification report's other criticals (R-2: nested CV, feature
selection in folds, panel-wide hyperparameters, data snapshot,
economic backtest, statistical inference) remain open and are tracked
on their own branches (`qf-04` through `qf-08`). They are not
addressed here.

For a CUDA host wanting to actually benefit from this fix, the
required environment is:

- `torch` with a CUDA-enabled wheel (current host has `2.10.0+cpu`).
- An installed CUDA toolkit + driver compatible with the torch wheel.
- For LightGBM GPU: a wheel built with `LIGHTGBM_GPU=1`. Stock PyPI
  wheels are CPU-only; the helper detects this and falls back
  transparently rather than failing mid-training.

Once on such a host, `python scripts/check_gpu.py` should report
`device='cuda'` / `task_type='GPU'` / `device_type='gpu'`, and any
`tune_all_models(...)` call with `CONFIG["use_gpu"]=True` will route
its inner training through the GPU.
