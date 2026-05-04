"""Leakage-safe nested Purged K-Fold pipeline.

This module is the canonical research pipeline. The legacy single-loop
helpers in :mod:`src.analysis` (``run_kfold_comparison``,
``run_detailed_single_stock_analysis``, ``run_all_stocks_purged_cv``,
``run_portfolio_analysis``) are retained for backward compatibility and
pedagogical comparison but **must not** be used to produce headline
numbers — they perform feature selection and Optuna tuning *outside*
the outer cross-validation, which the audit (C-2 / C-3 / C-4) flags as
the dominant source of upward bias in the previously reported 73%.

The contract enforced here
--------------------------

For every outer fold ``k``, every operation that consumes labels —
correlation filter, Boruta, inner Optuna tuning, model fit — sees only

::

    train_k = D \\ (test_k ∪ embargo_k ∪ purge_k)

The outer test fold is touched only by ``predict()`` once a final model
has been chosen and fitted on outer-train. Out-of-fold (OOF)
predictions are concatenated across outer folds and reported as both
per-fold metrics and a single pooled OOF metric (López de Prado's
"all-fold accuracy", *Advances in Financial Machine Learning*, ch. 7).

Audit cross-references
----------------------
- C-2 (no nested CV) → outer/inner Purged K-Fold structure below.
- C-3 (selection outside folds) → ``correlation_filter`` and
  ``boruta_selection`` are called only on the outer-train slice.
- C-4 (panel-wide hyperparameters) → ``tune_model`` runs per fold per
  asset; no parameter survives across folds.

Defaults
--------
- ``label_mode="raw_return"`` — causal labels (audit C-1 fix).
- ``include_changes=True``  — adds Δ-features alongside levels.
- ``feature_pool="extended"`` — full 14-indicator set.
- ``use_gpu`` is read from ``CONFIG`` and propagates through the
  helpers in :mod:`src.gpu` (audit C-18 / R-4 fix).

Wavelet labels are available via ``label_mode="wavelet"`` for
robustness only and are explicitly *not* the default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import CONFIG, MODEL_NAMES, TREE_MODEL_NAMES
from .feature_selection import boruta_selection, correlation_filter
from .models import (
    _train_predict,
    calculate_metrics,
    create_models,
)
from .preprocessing import DEFAULT_LABEL_MODE, prepare_features_with_t1
from .tuning import tune_model
from .validation import PurgedKFold


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NestedCVResult:
    """Bundle of artefacts produced by :func:`run_nested_purged_cv`.

    ``oof_predictions`` is the canonical headline-source — every row in
    the original ``X`` appears at most once and was predicted by a model
    that did not see it during training, selection, or tuning.
    """

    ticker: str
    window: int
    label_mode: str
    n_outer_splits: int
    n_inner_splits: int
    pct_embargo: float
    n_trials: int
    seed: int
    config_snapshot: dict

    oof_predictions: pd.DataFrame
    per_fold_metrics: pd.DataFrame
    pooled_metrics: pd.DataFrame
    selected_features: pd.DataFrame
    best_params: pd.DataFrame


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_nested_purged_cv(
    df: pd.DataFrame,
    *,
    ticker: str,
    window: int,
    config: Optional[dict] = None,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    pct_embargo: float = 0.01,
    n_trials: int = 20,
    model_names: Optional[Sequence[str]] = None,
    label_mode: str = DEFAULT_LABEL_MODE,
    include_changes: bool = True,
    feature_pool: str = "extended",
    corr_threshold: float = 0.9,
    boruta_max_iter: int = 100,
    seed: int = 42,
    verbose: bool = True,
    progress: bool = False,
) -> NestedCVResult:
    """Run a leakage-safe nested Purged K-Fold for one (ticker, window).

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV indexed by trading day.
    ticker : str
        Asset identifier (used for tagging the result frames).
    window : int
        Forecasting horizon ``h`` in trading days.
    n_outer_splits, n_inner_splits : int
        Outer (evaluation) and inner (tuning) Purged-KFold split counts.
    n_trials : int
        Optuna trials per (outer fold, model). Set low for smoke runs.
    model_names : sequence[str], optional
        Subset of ``TREE_MODEL_NAMES``; default = all five tree models.
    label_mode : str
        Default ``"raw_return"``. Pass ``"wavelet"`` only as a
        robustness check; it is non-causal (audit C-1).
    feature_pool, include_changes, corr_threshold, boruta_max_iter
        Forwarded to :func:`src.preprocessing.prepare_features_with_t1`
        and the feature-selection helpers.
    seed : int
        Random state propagated to feature-selection, Optuna, and model
        constructors. Stored in ``config_snapshot``.
    verbose : bool
        Print per-fold progress to stdout.

    Returns
    -------
    NestedCVResult
    """
    if config is None:
        config = {**CONFIG}
    else:
        config = {**config}
    config["random_state"] = seed
    if model_names is None:
        model_names = list(TREE_MODEL_NAMES)
    model_names = list(model_names)

    extended = feature_pool == "extended"

    # ------------------------------------------------------------------
    # Build the full feature matrix once, with the chosen label mode.
    # ------------------------------------------------------------------
    X_all, y_all, t1_all = prepare_features_with_t1(
        df, window=window, config=config,
        extended=extended,
        label_mode=label_mode,
        include_changes=include_changes,
    )

    if len(X_all) < n_outer_splits * 3:
        raise ValueError(
            f"Too few samples ({len(X_all)}) for {n_outer_splits} outer folds."
        )

    outer_pkf = PurgedKFold(
        n_splits=n_outer_splits, t1=t1_all, pct_embargo=pct_embargo,
    )

    oof_rows: list[dict] = []
    fold_metrics_rows: list[dict] = []
    selected_features_rows: list[dict] = []
    best_params_rows: list[dict] = []

    t_start_total = time.time()

    for outer_fold_id, (outer_train_idx, outer_test_idx) in enumerate(outer_pkf.split(X_all)):
        t_start_fold = time.time()
        if verbose:
            print(
                f"[{ticker} h={window}] outer fold {outer_fold_id + 1}/{n_outer_splits} "
                f"train={len(outer_train_idx)} test={len(outer_test_idx)}"
            )

        # ------------------------------------------------------------------
        # Slice the outer-train portion. The outer-test portion (X_te, y_te)
        # is held back and only used inside _train_predict at the very end
        # for prediction — never for fit, selection, or tuning.
        # ------------------------------------------------------------------
        X_tr = X_all.iloc[outer_train_idx]
        y_tr = y_all.iloc[outer_train_idx]
        t1_tr = t1_all.iloc[outer_train_idx]
        X_te = X_all.iloc[outer_test_idx]
        y_te = y_all.iloc[outer_test_idx]

        # ------------------------------------------------------------------
        # 1) Correlation filter on outer-train.
        # ------------------------------------------------------------------
        corr_selected, _, corr_dropped = correlation_filter(
            X_tr, threshold=corr_threshold,
        )
        for f in corr_dropped:
            selected_features_rows.append({
                "ticker": ticker, "outer_fold": outer_fold_id,
                "stage": "corr_dropped", "feature": f,
            })
        for f in corr_selected:
            selected_features_rows.append({
                "ticker": ticker, "outer_fold": outer_fold_id,
                "stage": "corr_kept", "feature": f,
            })

        # ------------------------------------------------------------------
        # 2) Boruta on the corr-filtered outer-train. Fall back gracefully
        #    if Boruta returns nothing or raises (small samples / pathology).
        # ------------------------------------------------------------------
        t_boruta = time.time()
        try:
            confirmed, tentative, _, _ = boruta_selection(
                X_tr[corr_selected], y_tr,
                random_state=seed, max_iter=boruta_max_iter,
            )
        except Exception as exc:  # pragma: no cover — defensive
            if verbose:
                print(f"  Boruta raised in fold {outer_fold_id}: {exc!r}; "
                      f"falling back to correlation-selected features.")
            confirmed = list(corr_selected)
            tentative = []

        if not confirmed:
            confirmed = list(tentative) if tentative else list(corr_selected)
            if verbose:
                print(f"  Boruta confirmed 0 features in fold "
                      f"{outer_fold_id}; using {len(confirmed)} fallback.")
        if progress:
            print(f"  Boruta: {len(confirmed)} confirmed in "
                  f"{time.time() - t_boruta:.1f}s")

        for f in confirmed:
            selected_features_rows.append({
                "ticker": ticker, "outer_fold": outer_fold_id,
                "stage": "final", "feature": f,
            })

        X_tr_sel = X_tr[confirmed]
        X_te_sel = X_te[confirmed]

        # ------------------------------------------------------------------
        # 3) Inner Optuna tuning per model. Each call uses a *restricted*
        #    PurgedKFold whose t1 is t1_tr (only outer-train timestamps),
        #    so inner purging cannot leak from outer-test.
        # ------------------------------------------------------------------
        per_model_best: dict[str, dict] = {}
        for model_name in model_names:
            t_tune = time.time()
            tuning_result = tune_model(
                model_name,
                X_tr_sel, y_tr, t1_tr,
                n_trials=n_trials,
                n_splits=n_inner_splits,
                pct_embargo=pct_embargo,
                config=config,
            )
            per_model_best[model_name] = tuning_result["best_params"]
            if progress:
                best_score = tuning_result.get(
                    "best_accuracy", tuning_result.get("best_score", float("nan"))
                )
                print(f"  Tuning [{model_name}]: best_acc={best_score:.4f} "
                      f"in {time.time() - t_tune:.1f}s ({n_trials} trials)")
            for k, v in tuning_result["best_params"].items():
                best_params_rows.append({
                    "ticker": ticker,
                    "outer_fold": outer_fold_id,
                    "model": model_name,
                    "param": k,
                    "value": v,
                })

        # ------------------------------------------------------------------
        # 4) Build models with the per-fold tuned hyperparameters and
        #    fit on the outer-train slice. ``create_models`` reads
        #    ``config["use_gpu"]`` and applies GPU kwargs via src.gpu —
        #    no extra plumbing needed here.
        # ------------------------------------------------------------------
        models = create_models(config=config, hyperparams=per_model_best)

        t_final = time.time()
        for model_name in model_names:
            if model_name not in models:
                continue
            model = models[model_name]
            _, y_pred_bin, y_proba = _train_predict(model, X_tr_sel, y_tr, X_te_sel)
            y_pred_signed = (np.asarray(y_pred_bin) * 2 - 1).astype(int)

            # Out-of-fold predictions
            for i, idx in enumerate(X_te_sel.index):
                proba = float(y_proba[i]) if y_proba is not None else np.nan
                oof_rows.append({
                    "ticker": ticker,
                    "outer_fold": outer_fold_id,
                    "date": idx,
                    "window": window,
                    "model": model_name,
                    "y_true": int(y_te.iloc[i]),
                    "y_pred": int(y_pred_signed[i]),
                    "y_proba": proba,
                })

            metrics = calculate_metrics(y_te, y_pred_signed, y_proba)
            fold_metrics_rows.append({
                "ticker": ticker,
                "outer_fold": outer_fold_id,
                "window": window,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f_score"],
                "auc": metrics["auc"],
                "n_test": int(len(y_te)),
            })

        if progress:
            print(f"  Final fit + predict (all models): "
                  f"{time.time() - t_final:.1f}s")
        if verbose:
            fold_elapsed = time.time() - t_start_fold
            elapsed_total = time.time() - t_start_total
            folds_done = outer_fold_id + 1
            folds_left = n_outer_splits - folds_done
            avg_fold = elapsed_total / folds_done
            eta_s = avg_fold * folds_left
            eta_m, eta_sec = divmod(int(eta_s), 60)
            print(f"  fold {folds_done}/{n_outer_splits} done in "
                  f"{fold_elapsed:.1f}s; cumulative {elapsed_total:.0f}s; "
                  f"ETA {eta_m}m{eta_sec:02d}s "
                  f"({folds_left} folds remaining)")

    oof_df = pd.DataFrame(oof_rows)
    fold_df = pd.DataFrame(fold_metrics_rows)
    selected_df = pd.DataFrame(selected_features_rows)
    bp_df = pd.DataFrame(best_params_rows)

    # ------------------------------------------------------------------
    # Pooled OOF metrics: one accuracy / AUC / etc. per model, computed
    # over the full set of OOF predictions (not the average of per-fold
    # accuracies). This is the headline number.
    # ------------------------------------------------------------------
    pooled_rows: list[dict] = []
    for model_name in model_names:
        sub = oof_df[oof_df["model"] == model_name]
        if sub.empty:
            continue
        y_true_all = sub["y_true"].astype(int).values
        y_pred_all = sub["y_pred"].astype(int).values
        proba_arr = sub["y_proba"].astype(float).values
        proba_arr = proba_arr if not np.all(np.isnan(proba_arr)) else None
        m = calculate_metrics(pd.Series(y_true_all), y_pred_all, proba_arr)
        pooled_rows.append({
            "ticker": ticker,
            "window": window,
            "model": model_name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f_score"],
            "auc": m["auc"],
            "n_test_total": int(len(sub)),
        })
    pooled_df = pd.DataFrame(pooled_rows)

    config_snapshot = {
        "ticker": ticker,
        "window": window,
        "label_mode": label_mode,
        "include_changes": include_changes,
        "feature_pool": feature_pool,
        "n_outer_splits": n_outer_splits,
        "n_inner_splits": n_inner_splits,
        "pct_embargo": pct_embargo,
        "n_trials": n_trials,
        "model_names": list(model_names),
        "seed": seed,
        "corr_threshold": corr_threshold,
        "boruta_max_iter": boruta_max_iter,
        "use_gpu": config.get("use_gpu", False),
        "prefer_gpu": config.get("prefer_gpu", True),
        "n_samples_total": int(len(X_all)),
        "wall_seconds": round(time.time() - t_start_total, 2),
    }

    if verbose:
        print(f"[{ticker} h={window}] total wall time: "
              f"{config_snapshot['wall_seconds']:.1f}s")

    return NestedCVResult(
        ticker=ticker, window=window, label_mode=label_mode,
        n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits,
        pct_embargo=pct_embargo, n_trials=n_trials, seed=seed,
        config_snapshot=config_snapshot,
        oof_predictions=oof_df,
        per_fold_metrics=fold_df,
        pooled_metrics=pooled_df,
        selected_features=selected_df,
        best_params=bp_df,
    )
