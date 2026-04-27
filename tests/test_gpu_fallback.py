"""Unit tests for the GPU helpers and the CPU-fallback contract.

These tests do **not** require a CUDA host. They use ``monkeypatch`` to
toggle ``src.gpu.cuda_available`` and verify that:

- when CUDA is unavailable, every backend helper returns CPU parameters;
- when CUDA *is* available **and** ``CONFIG["use_gpu"]`` is True, the
  factories return GPU parameters;
- when ``CONFIG["use_gpu"]`` is False, the factories return CPU
  parameters even on a CUDA host (asymmetric DL-vs-tree bug fix);
- LightGBM stays on CPU when its installed build does not support GPU,
  even with CUDA available.

Run with::

    python -m pytest tests/test_gpu_fallback.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.gpu as gpu_module  # noqa: E402
from src.config import CONFIG  # noqa: E402
from src.gpu import (  # noqa: E402
    get_catboost_gpu_params,
    get_lightgbm_gpu_params,
    get_torch_device,
    get_xgboost_gpu_params,
)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Pretend a CUDA device is available, with LightGBM GPU build allowed."""
    monkeypatch.setattr(gpu_module, "cuda_available", lambda: True)
    monkeypatch.setattr(gpu_module, "_LIGHTGBM_GPU_PROBE_CACHE", True)
    yield


@pytest.fixture
def fake_no_cuda(monkeypatch):
    monkeypatch.setattr(gpu_module, "cuda_available", lambda: False)
    monkeypatch.setattr(gpu_module, "_LIGHTGBM_GPU_PROBE_CACHE", False)
    yield


# ---------------------------------------------------------------------------
# Helper return-value contracts
# ---------------------------------------------------------------------------

def test_get_torch_device_cpu_when_no_cuda(fake_no_cuda):
    assert get_torch_device(prefer_gpu=True) == "cpu"
    assert get_torch_device(prefer_gpu=False) == "cpu"


def test_get_torch_device_cuda_only_when_preferred(fake_cuda):
    assert get_torch_device(prefer_gpu=True) == "cuda"
    # prefer_gpu=False means the caller wants CPU even on a CUDA host.
    assert get_torch_device(prefer_gpu=False) == "cpu"


def test_xgboost_gpu_helper_falls_back(fake_no_cuda):
    p = get_xgboost_gpu_params(use_gpu=True)
    assert p["device"] == "cpu"
    assert p["tree_method"] == "hist"


def test_xgboost_gpu_helper_picks_cuda(fake_cuda):
    p = get_xgboost_gpu_params(use_gpu=True)
    assert p["device"] == "cuda"
    p_off = get_xgboost_gpu_params(use_gpu=False)
    assert p_off["device"] == "cpu", "use_gpu=False must force CPU"


def test_catboost_gpu_helper_falls_back(fake_no_cuda):
    p = get_catboost_gpu_params(use_gpu=True)
    assert p["task_type"] == "CPU"
    # bootstrap_type must NOT be injected on CPU (preserves CPU defaults).
    assert "bootstrap_type" not in p


def test_catboost_gpu_helper_picks_gpu(fake_cuda):
    p = get_catboost_gpu_params(use_gpu=True)
    assert p["task_type"] == "GPU"
    # Bernoulli is required so the existing `subsample` argument keeps working.
    assert p["bootstrap_type"] == "Bernoulli"


def test_lightgbm_helper_falls_back_when_build_lacks_gpu(monkeypatch):
    # Even on a CUDA host, if the installed LightGBM was built CPU-only
    # the helper must keep us on CPU.
    monkeypatch.setattr(gpu_module, "cuda_available", lambda: True)
    monkeypatch.setattr(gpu_module, "_LIGHTGBM_GPU_PROBE_CACHE", False)
    p = get_lightgbm_gpu_params(use_gpu=True)
    assert p["device_type"] == "cpu"


def test_lightgbm_helper_picks_gpu_when_supported(fake_cuda):
    p = get_lightgbm_gpu_params(use_gpu=True)
    assert p["device_type"] == "gpu"
    assert get_lightgbm_gpu_params(use_gpu=False)["device_type"] == "cpu"


# ---------------------------------------------------------------------------
# Factory wiring — CONFIG["use_gpu"] must be respected by tree AND DL paths
# ---------------------------------------------------------------------------

def _cfg(**overrides):
    return {**CONFIG, **overrides}


def test_create_models_respects_use_gpu_off(fake_cuda):
    from src.models import create_models

    m = create_models(_cfg(use_gpu=False))
    assert getattr(m["XGBoost"], "device", None) == "cpu"
    assert m["CatBoost"].get_param("task_type") == "CPU"
    assert m["LightGBM"].get_params().get("device_type") == "cpu"


def test_create_models_uses_gpu_when_on(fake_cuda):
    from src.models import create_models

    m = create_models(_cfg(use_gpu=True))
    assert getattr(m["XGBoost"], "device", None) == "cuda"
    assert m["CatBoost"].get_param("task_type") == "GPU"
    assert m["LightGBM"].get_params().get("device_type") == "gpu"


def test_create_dl_models_respects_use_gpu_off(fake_cuda):
    """Regression test: previously create_dl_models hardcoded prefer_gpu=True
    inside the wrappers and ignored CONFIG["use_gpu"]. Verified by simulating
    a CUDA host and asserting that the resolved device stays on CPU when the
    operator opts out via config."""
    from src.models import create_dl_models

    dl = create_dl_models(_cfg(use_gpu=False), input_dim=6)
    assert dl["MLP"].device == "cpu"
    assert dl["LSTM"].device == "cpu"
    assert dl["MLP"]._resolve_device() == "cpu"
    assert dl["LSTM"]._resolve_device() == "cpu"


def test_create_dl_models_uses_gpu_when_on(fake_cuda):
    from src.models import create_dl_models

    dl = create_dl_models(_cfg(use_gpu=True), input_dim=6)
    assert dl["MLP"].device == "cuda"
    assert dl["LSTM"].device == "cuda"


# ---------------------------------------------------------------------------
# Sklearn-only models stay CPU-only
# ---------------------------------------------------------------------------

def test_random_forest_has_no_gpu_kwargs(fake_cuda):
    """sklearn RandomForestClassifier has no native CUDA path; the model
    factory must not inject any GPU parameters into it."""
    from src.models import create_models

    rf = create_models(_cfg(use_gpu=True))["Random Forest"]
    params = rf.get_params()
    assert "device" not in params
    assert "task_type" not in params
    assert "device_type" not in params


def test_sklearn_gradient_boosting_has_no_gpu_kwargs(fake_cuda):
    from src.models import create_models

    gb = create_models(_cfg(use_gpu=True))["Gradient Boosting"]
    params = gb.get_params()
    assert "device" not in params
    assert "task_type" not in params
    assert "device_type" not in params


# ---------------------------------------------------------------------------
# tuning.py — GPU wiring (R-4 regression tests)
# ---------------------------------------------------------------------------

class _StubTrial:
    """Minimal Optuna trial replacement: returns midpoint / first-option for
    every suggest_* call so the objective just builds *some* valid model."""
    def suggest_int(self, name, lo, hi):
        return (lo + hi) // 2

    def suggest_float(self, name, lo, hi, log=False):
        return (lo + hi) / 2.0

    def suggest_categorical(self, name, options):
        return options[0]


def _run_objectives(monkeypatch, cfg):
    """Drive _xgb / _lgb / _catboost objectives with a stubbed _cv_score
    that captures the constructed model and returns a constant score."""
    import numpy as np
    import pandas as pd
    from src import tuning as tuning_mod

    rng = np.random.default_rng(0)
    n = 80
    idx = pd.bdate_range("2020-01-01", periods=n)
    X = pd.DataFrame(rng.standard_normal((n, 4)), index=idx, columns=list("abcd"))
    y = pd.Series(rng.integers(0, 2, size=n) * 2 - 1, index=idx, dtype="int64")
    t1 = pd.Series(idx.shift(1).fillna(idx[-1]), index=idx)

    captured = []
    monkeypatch.setattr(
        tuning_mod, "_cv_score",
        lambda model, *args, **kwargs: (captured.append(model) or 0.5),
    )
    tuning_mod._xgb_objective(_StubTrial(), X, y, t1, 3, 0.0, cfg)
    tuning_mod._lgb_objective(_StubTrial(), X, y, t1, 3, 0.0, cfg)
    tuning_mod._catboost_objective(_StubTrial(), X, y, t1, 3, 0.0, cfg)
    return captured  # [xgb, lgb, catboost]


def test_tuning_objectives_apply_gpu_kwargs(fake_cuda, monkeypatch):
    """R-4: when use_gpu=True on a CUDA host, every Optuna objective must
    construct GPU-configured models — same kwargs as create_models."""
    cfg = {**CONFIG, "use_gpu": True}
    xgb_m, lgb_m, cb_m = _run_objectives(monkeypatch, cfg)

    assert getattr(xgb_m, "device", None) == "cuda"
    assert lgb_m.get_params().get("device_type") == "gpu"
    assert cb_m.get_param("task_type") == "GPU"
    assert cb_m.get_param("bootstrap_type") == "Bernoulli"


def test_tuning_objectives_respect_use_gpu_off(fake_cuda, monkeypatch):
    """R-4: setting use_gpu=False must keep every Optuna objective on CPU
    even when CUDA is available."""
    cfg = {**CONFIG, "use_gpu": False}
    xgb_m, lgb_m, cb_m = _run_objectives(monkeypatch, cfg)

    assert getattr(xgb_m, "device", None) == "cpu"
    assert lgb_m.get_params().get("device_type") == "cpu"
    assert cb_m.get_param("task_type") == "CPU"


# ---------------------------------------------------------------------------
# CPU host: real environment must work without CUDA
# ---------------------------------------------------------------------------

def test_real_cpu_host_smoke():
    """On the actual host (which may or may not have CUDA), every helper
    must return a valid kwargs dict and the factories must instantiate
    without error. This is the pure-fallback smoke test."""
    from src.models import create_models, create_dl_models

    m = create_models()
    dl = create_dl_models(input_dim=6)
    assert set(m.keys()) == {
        "Random Forest", "XGBoost", "Gradient Boosting", "LightGBM", "CatBoost"
    }
    assert set(dl.keys()) == {"MLP", "LSTM"}
    # Whichever device the host picks, it must be one of the two known values.
    assert dl["MLP"]._resolve_device() in ("cpu", "cuda")
