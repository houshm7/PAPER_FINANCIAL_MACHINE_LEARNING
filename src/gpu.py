"""GPU detection and backend-specific GPU/CPU parameter helpers.

All probes are best-effort and never raise to the caller: if any backend
import or capability check fails, the helper returns CPU parameters.

Conventions
-----------
- ``cuda_available()`` is the single source of truth for "is there a usable
  CUDA device?" and is built on ``torch.cuda.is_available()``. Other backends
  reuse this gate before attempting GPU configurations.
- Each ``get_<backend>_gpu_params(use_gpu=True)`` returns a kwargs dict that
  can be unpacked directly into the backend's classifier constructor.
- LightGBM is special-cased: stock pip wheels are CPU-only, so we both
  require ``use_gpu=True`` *and* probe a tiny LightGBM training run with
  ``device_type="gpu"`` once. The probe result is cached.

This module imports ``torch`` unconditionally (the project already depends on
it for ``deep_learning.py``); other backend imports are local to keep this
file safe to import even if a backend is missing.
"""

from __future__ import annotations

import os
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Core CUDA gate
# ---------------------------------------------------------------------------

def cuda_available() -> bool:
    """Return True if a CUDA device is usable through PyTorch."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_device_name(index: int = 0) -> str:
    """Return the name of CUDA device ``index`` or ``""`` when unavailable."""
    if not cuda_available():
        return ""
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return ""


def get_torch_device(prefer_gpu: bool = True) -> str:
    """Return ``"cuda"`` when requested and available, else ``"cpu"``."""
    if prefer_gpu and cuda_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def get_xgboost_gpu_params(use_gpu: bool = True) -> dict[str, Any]:
    """Return XGBoost kwargs for GPU (when available) or CPU.

    Uses the modern XGBoost ≥ 2.0 ``device`` argument. ``tree_method="hist"``
    is the recommended setting on both CPU and GPU.
    """
    if use_gpu and cuda_available():
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "hist", "device": "cpu"}


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def get_catboost_gpu_params(use_gpu: bool = True) -> dict[str, Any]:
    """Return CatBoost kwargs for GPU (when available) or CPU.

    On GPU the default ``bootstrap_type`` is ``"Bayesian"``, which is
    incompatible with the ``subsample`` argument used elsewhere in the
    pipeline. We force ``"Bernoulli"`` on GPU so subsampling semantics
    stay as close to the CPU path as possible. CPU users see no change.
    """
    if use_gpu and cuda_available():
        return {
            "task_type": "GPU",
            "devices": "0",
            "bootstrap_type": "Bernoulli",
        }
    return {"task_type": "CPU"}


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

_LIGHTGBM_GPU_PROBE_CACHE: bool | None = None


def lightgbm_gpu_supported() -> bool:
    """Return True only if the installed LightGBM build supports GPU.

    Many wheels on PyPI are compiled CPU-only; instantiating one of them with
    ``device_type="gpu"`` raises an opaque error mid-training. We probe once
    by training a tiny model and cache the result for the rest of the
    process.
    """
    global _LIGHTGBM_GPU_PROBE_CACHE
    if _LIGHTGBM_GPU_PROBE_CACHE is not None:
        return _LIGHTGBM_GPU_PROBE_CACHE
    if not cuda_available():
        _LIGHTGBM_GPU_PROBE_CACHE = False
        return False

    try:
        import numpy as np
        import lightgbm as lgb

        rng = np.random.default_rng(0)
        X = rng.standard_normal((32, 4)).astype("float32")
        y = (rng.standard_normal(32) > 0).astype("int32")
        booster = lgb.LGBMClassifier(
            n_estimators=1,
            num_leaves=3,
            device_type="gpu",
            verbose=-1,
        )
        booster.fit(X, y)
    except Exception:
        _LIGHTGBM_GPU_PROBE_CACHE = False
        return False

    _LIGHTGBM_GPU_PROBE_CACHE = True
    return True


def get_lightgbm_gpu_params(use_gpu: bool = True) -> dict[str, Any]:
    """Return LightGBM kwargs for GPU only when the installed build supports it.

    Falls back to CPU when (a) ``use_gpu`` is False, (b) there is no CUDA
    device, or (c) the LightGBM build was not compiled with GPU support.
    """
    if use_gpu and lightgbm_gpu_supported():
        return {"device_type": "gpu"}
    return {"device_type": "cpu"}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def gpu_summary() -> dict[str, Any]:
    """Return a structured summary of GPU support for logging / scripts."""
    info: dict[str, Any] = {
        "cuda_available": cuda_available(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device_name": cuda_device_name(0) if cuda_available() else "",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    info["xgboost_params"] = get_xgboost_gpu_params(use_gpu=True)
    info["catboost_params"] = get_catboost_gpu_params(use_gpu=True)
    info["lightgbm_gpu_supported"] = lightgbm_gpu_supported()
    info["lightgbm_params"] = get_lightgbm_gpu_params(use_gpu=True)
    return info
