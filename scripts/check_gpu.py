"""Diagnostic script: report GPU availability for every backend used here.

Run with::

    python scripts/check_gpu.py

The script trains a tiny model on each backend with the GPU configuration
that ``src.gpu`` would select, prints a per-backend status line, and exits 0
even when nothing is GPU-capable. The point is to surface mis-installed
CUDA/cuDNN, CPU-only LightGBM wheels, or missing CatBoost GPU support
*before* a 30-minute experiment fails halfway through.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repository root importable when invoked as ``python scripts/check_gpu.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.gpu import (
    cuda_available,
    cuda_device_name,
    get_catboost_gpu_params,
    get_lightgbm_gpu_params,
    get_torch_device,
    get_xgboost_gpu_params,
    gpu_summary,
    lightgbm_gpu_supported,
)


def _line(label: str, ok: bool, detail: str = "") -> None:
    mark = "OK" if ok else "--"
    sep = " | " if detail else ""
    print(f"  [{mark}] {label}{sep}{detail}")


def _check_torch() -> None:
    print("PyTorch / CUDA")
    print(f"  cuda_available(): {cuda_available()}")
    if cuda_available():
        print(f"  device name:      {cuda_device_name(0)}")
    print(f"  selected device:  {get_torch_device(prefer_gpu=True)}")


def _train_xgboost() -> None:
    print("XGBoost")
    try:
        import xgboost as xgb
    except Exception as exc:
        _line("import", False, repr(exc))
        return
    params = get_xgboost_gpu_params(use_gpu=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, 6)).astype("float32")
    y = (rng.standard_normal(128) > 0).astype("int32")
    try:
        t0 = time.time()
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, **params)
        model.fit(X, y)
        dt = time.time() - t0
        _line(f"params={params}", True, f"fit={dt:.2f}s")
    except Exception as exc:
        _line(f"params={params}", False, repr(exc))


def _train_catboost() -> None:
    print("CatBoost")
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        _line("import", False, repr(exc))
        return
    params = get_catboost_gpu_params(use_gpu=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, 6)).astype("float32")
    y = (rng.standard_normal(128) > 0).astype("int32")
    try:
        t0 = time.time()
        model = CatBoostClassifier(iterations=10, depth=3, verbose=False, **params)
        model.fit(X, y)
        dt = time.time() - t0
        _line(f"params={params}", True, f"fit={dt:.2f}s")
    except Exception as exc:
        _line(f"params={params}", False, repr(exc))


def _train_lightgbm() -> None:
    print("LightGBM")
    try:
        import lightgbm as lgb
    except Exception as exc:
        _line("import", False, repr(exc))
        return
    print(f"  gpu_supported probe: {lightgbm_gpu_supported()}")
    params = get_lightgbm_gpu_params(use_gpu=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, 6)).astype("float32")
    y = (rng.standard_normal(128) > 0).astype("int32")
    try:
        t0 = time.time()
        model = lgb.LGBMClassifier(n_estimators=10, num_leaves=7, verbose=-1, **params)
        model.fit(X, y)
        dt = time.time() - t0
        _line(f"params={params}", True, f"fit={dt:.2f}s")
    except Exception as exc:
        _line(f"params={params}", False, repr(exc))


def _train_torch_mlp() -> None:
    print("PyTorch MLP (skorch)")
    try:
        from src.deep_learning import SklearnMLPClassifier
    except Exception as exc:
        _line("import", False, repr(exc))
        return
    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, 6)).astype("float32")
    y = (rng.standard_normal(128) > 0).astype("int64")
    try:
        t0 = time.time()
        model = SklearnMLPClassifier(
            input_dim=6, hidden_dim=8, n_layers=1,
            max_epochs=2, batch_size=32,
        )
        model.fit(X, y)
        dt = time.time() - t0
        _line(f"device={model._resolve_device()}", True, f"fit={dt:.2f}s")
    except Exception as exc:
        _line("fit", False, repr(exc))


def main() -> int:
    print("=" * 60)
    print("GPU support diagnostic")
    print("=" * 60)

    _check_torch()
    print()
    _train_xgboost()
    print()
    _train_catboost()
    print()
    _train_lightgbm()
    print()
    _train_torch_mlp()
    print()

    print("Summary dict (from src.gpu.gpu_summary):")
    for k, v in gpu_summary().items():
        print(f"  {k:>26}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
