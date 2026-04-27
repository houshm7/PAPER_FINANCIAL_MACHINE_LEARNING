"""Leakage-contract tests for the nested Purged K-Fold pipeline.

Each test enforces one half of the contract from
``docs/audit/06_nested_cv_report.md``:

1. Feature selection (``correlation_filter`` + ``boruta_selection``)
   only sees outer-train indices for its fold.
2. Optuna inner tuning (``tune_model``) only sees outer-train indices
   for its fold.
3. ``X``, ``y``, ``t1`` share an index after preparation.
4. The outer-test fold is touched only by the final ``predict()``;
   selection and tuning never see its observations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.pipeline as pipeline_mod  # noqa: E402
from src.pipeline import run_nested_purged_cv  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic OHLCV
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, size=n)
    # Inject mild AR(1) so something is learnable.
    for t in range(1, n):
        rets[t] += 0.10 * rets[t - 1]
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0, 0.01, size=n))
    low = close * (1.0 - rng.uniform(0, 0.01, size=n))
    open_ = close * (1.0 + rng.normal(0, 0.002, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Shared smoke run with monkey-patched probes
# ---------------------------------------------------------------------------

@pytest.fixture
def probed_run(monkeypatch):
    """Run a tiny nested CV with probes on every label-consuming helper.

    Returns a dict with:
      - result            : NestedCVResult
      - corr_indices      : list of frozensets seen by correlation_filter
      - boruta_indices    : list of frozensets seen by boruta_selection
      - tune_indices      : list of frozensets seen by tune_model
      - tune_models       : list of model names seen by tune_model
      - test_dates        : Series mapping outer_fold -> set of test dates
    """
    corr_indices: list[frozenset] = []
    boruta_indices: list[frozenset] = []
    tune_indices: list[frozenset] = []
    tune_models: list[str] = []

    real_corr = pipeline_mod.correlation_filter
    real_boruta = pipeline_mod.boruta_selection
    real_tune = pipeline_mod.tune_model

    def fake_corr(X, threshold=0.9):
        corr_indices.append(frozenset(X.index))
        return real_corr(X, threshold=threshold)

    def fake_boruta(X, y, random_state=42, max_iter=100):
        boruta_indices.append(frozenset(X.index))
        return real_boruta(X, y, random_state=random_state, max_iter=max_iter)

    def fake_tune(model_name, X, y, t1, **kwargs):
        tune_models.append(model_name)
        tune_indices.append(frozenset(X.index))
        return real_tune(model_name, X, y, t1, **kwargs)

    monkeypatch.setattr(pipeline_mod, "correlation_filter", fake_corr)
    monkeypatch.setattr(pipeline_mod, "boruta_selection", fake_boruta)
    monkeypatch.setattr(pipeline_mod, "tune_model", fake_tune)

    df = _ohlcv(n=240, seed=11)
    result = run_nested_purged_cv(
        df, ticker="TEST", window=1,
        n_outer_splits=3, n_inner_splits=2,
        n_trials=2, model_names=["XGBoost"],
        boruta_max_iter=10, verbose=False,
    )
    test_dates = (
        result.oof_predictions
        .groupby("outer_fold")["date"]
        .apply(lambda s: frozenset(pd.to_datetime(s).tolist()))
    )
    return {
        "result": result,
        "corr_indices": corr_indices,
        "boruta_indices": boruta_indices,
        "tune_indices": tune_indices,
        "tune_models": tune_models,
        "test_dates": test_dates,
    }


# ---------------------------------------------------------------------------
# 1. Feature selection scope
# ---------------------------------------------------------------------------

def test_feature_selection_uses_outer_train_only(probed_run):
    """C-3: correlation_filter and boruta_selection must, per outer fold,
    see only indices that lie in that fold's outer-train slice — i.e.,
    they must be disjoint with that fold's own outer-test dates.

    They MAY include other folds' test dates (those are training data
    when working on the current fold). The check is strictly per-fold.
    """
    n_outer = probed_run["result"].n_outer_splits
    test_dates = probed_run["test_dates"]
    assert len(probed_run["corr_indices"]) == n_outer
    assert len(probed_run["boruta_indices"]) == n_outer

    for fold_id in range(n_outer):
        own_test = test_dates.loc[fold_id]
        assert own_test.isdisjoint(probed_run["corr_indices"][fold_id]), (
            f"correlation_filter saw fold {fold_id}'s test dates"
        )
        assert own_test.isdisjoint(probed_run["boruta_indices"][fold_id]), (
            f"boruta_selection saw fold {fold_id}'s test dates"
        )


# ---------------------------------------------------------------------------
# 2. Tuning scope
# ---------------------------------------------------------------------------

def test_optuna_tuning_uses_outer_train_only(probed_run):
    """C-2: tune_model must, per outer fold, only see indices in that
    fold's outer-train slice."""
    n_outer = probed_run["result"].n_outer_splits
    n_models = len(set(probed_run["tune_models"]))
    expected = n_outer * n_models
    assert len(probed_run["tune_indices"]) == expected

    test_dates = probed_run["test_dates"]
    # tune_model is called once per (outer_fold, model). The order is
    # (fold 0, model_1), (fold 0, model_2), ..., (fold 1, model_1), ...
    for i, idxs in enumerate(probed_run["tune_indices"]):
        fold_id = i // n_models
        own_test = test_dates.loc[fold_id]
        assert own_test.isdisjoint(idxs), (
            f"tune_model({probed_run['tune_models'][i]}) saw fold "
            f"{fold_id}'s test dates"
        )


# ---------------------------------------------------------------------------
# 3. X / y / t1 alignment
# ---------------------------------------------------------------------------

def test_x_y_t1_alignment_after_preparation():
    """The pipeline relies on prepare_features_with_t1 to give back
    aligned (X, y, t1). This test pins the contract independently of
    the nested CV machinery (defence in depth)."""
    from src.preprocessing import prepare_features_with_t1
    df = _ohlcv(n=200, seed=3)
    X, y, t1 = prepare_features_with_t1(df, window=2)
    assert X.index.equals(y.index)
    assert X.index.equals(t1.index)
    assert y.notna().all()
    assert t1.notna().all()


def test_predictions_index_is_subset_of_full_x_index():
    """After running the pipeline, every predicted date must be a date
    that was in the prepared X — no fabricated rows, no shifted dates."""
    from src.preprocessing import prepare_features_with_t1
    df = _ohlcv(n=240, seed=4)
    X, _, _ = prepare_features_with_t1(df, window=1)
    result = run_nested_purged_cv(
        df, ticker="TEST", window=1,
        n_outer_splits=3, n_inner_splits=2, n_trials=1,
        model_names=["XGBoost"], boruta_max_iter=10, verbose=False,
    )
    pred_dates = pd.to_datetime(result.oof_predictions["date"]).unique()
    assert set(pred_dates).issubset(set(X.index))


# ---------------------------------------------------------------------------
# 4. Outer-test untouched until final prediction
# ---------------------------------------------------------------------------

def test_outer_test_fold_untouched_until_prediction(probed_run):
    """Combined property:
      (a) for each outer fold, every label-consuming call (corr filter,
          Boruta, Optuna tuning) is disjoint with that fold's test dates;
      (b) every outer-test date appears in the OOF predictions exactly
          once per model — i.e., the test fold IS predicted on, just
          never trained on.
    """
    n_outer = probed_run["result"].n_outer_splits
    n_models = len(set(probed_run["tune_models"]))
    test_dates = probed_run["test_dates"]

    # (a) per-fold disjointness across all label-consuming entry points
    for fold_id in range(n_outer):
        own = test_dates.loc[fold_id]
        seen_in_fold: set = set()
        seen_in_fold |= probed_run["corr_indices"][fold_id]
        seen_in_fold |= probed_run["boruta_indices"][fold_id]
        for j in range(fold_id * n_models, (fold_id + 1) * n_models):
            seen_in_fold |= probed_run["tune_indices"][j]
        assert own.isdisjoint(seen_in_fold), (
            f"fold {fold_id}: leakage — outer-test dates appeared in a "
            f"selection or tuning call for that fold"
        )

    # (b) every outer-test date is predicted exactly once per model
    oof = probed_run["result"].oof_predictions
    assert not oof.empty
    counts = (
        oof.groupby(["model", "date"])
        .size()
        .reset_index(name="n")
    )
    assert (counts["n"] == 1).all(), (
        "every (model, date) pair must have exactly one prediction"
    )


# ---------------------------------------------------------------------------
# Smoke: pipeline runs end-to-end and produces the four expected frames
# ---------------------------------------------------------------------------

def test_pipeline_smoke_end_to_end():
    df = _ohlcv(n=240, seed=5)
    result = run_nested_purged_cv(
        df, ticker="SMOKE", window=1,
        n_outer_splits=3, n_inner_splits=2,
        n_trials=2, model_names=["XGBoost", "Random Forest"],
        boruta_max_iter=10, verbose=False,
    )
    # Headline structures present and non-empty
    assert not result.oof_predictions.empty
    assert not result.per_fold_metrics.empty
    assert not result.pooled_metrics.empty
    assert not result.selected_features.empty
    assert not result.best_params.empty

    # Sanity columns
    assert {"ticker", "outer_fold", "date", "model",
            "y_true", "y_pred", "y_proba"}.issubset(result.oof_predictions.columns)
    assert {"ticker", "outer_fold", "model", "accuracy",
            "auc"}.issubset(result.per_fold_metrics.columns)
    assert {"ticker", "model", "accuracy"}.issubset(result.pooled_metrics.columns)
    assert {"ticker", "outer_fold", "stage",
            "feature"}.issubset(result.selected_features.columns)
    assert {"ticker", "outer_fold", "model",
            "param", "value"}.issubset(result.best_params.columns)

    # Pooled accuracy lies in [0, 1]
    assert (result.pooled_metrics["accuracy"] >= 0).all()
    assert (result.pooled_metrics["accuracy"] <= 1).all()
