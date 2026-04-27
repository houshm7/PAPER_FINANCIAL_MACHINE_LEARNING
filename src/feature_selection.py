"""Feature selection pipeline for stock market prediction.

Three-step approach:
  1. Correlation filter — remove redundant features (pairwise |ρ| > threshold)
  2. Boruta — statistically validate which features are truly important
  3. SHAP — interpret feature contributions (post-modelling)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import shap

from .config import CONFIG


# ---------------------------------------------------------------------------
# 1. Correlation Filter
# ---------------------------------------------------------------------------

def correlation_filter(X, threshold=0.9):
    """Remove features with pairwise correlation above threshold.

    Keeps the first feature in each correlated pair (order matters).

    Returns
    -------
    selected : list[str] — feature names to keep.
    corr_matrix : pd.DataFrame — full correlation matrix.
    dropped : list[str] — feature names removed.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    dropped = [col for col in upper.columns if any(upper[col] > threshold)]
    selected = [col for col in X.columns if col not in dropped]
    return selected, corr_matrix, dropped


def correlation_with_target(X, y):
    """Compute correlation between each feature and the target.

    Returns
    -------
    pd.Series — absolute correlation with target, sorted descending.
    """
    y_bin = (y == 1).astype(int)
    corr = X.corrwith(y_bin).abs().sort_values(ascending=False)
    return corr


# ---------------------------------------------------------------------------
# 2. Boruta
# ---------------------------------------------------------------------------

def boruta_selection(X, y, random_state=42, max_iter=100):
    """Select features confirmed as important by Boruta algorithm.

    Boruta creates 'shadow features' (randomized copies) and iteratively
    tests whether each real feature is significantly more important than
    the best shadow feature.

    Returns
    -------
    confirmed : list[str] — confirmed important features.
    tentative : list[str] — tentative features (borderline).
    rejected : list[str] — rejected features.
    ranking : pd.Series — Boruta ranking (1 = confirmed).
    """
    y_bin = (y == 1).astype(int)
    rf = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=-1
    )
    selector = BorutaPy(rf, n_estimators="auto", random_state=random_state, max_iter=max_iter)
    selector.fit(X.values, y_bin.values)

    ranking = pd.Series(selector.ranking_, index=X.columns)
    confirmed = list(X.columns[selector.support_])
    tentative = list(X.columns[selector.support_weak_])
    rejected = [c for c in X.columns if c not in confirmed and c not in tentative]
    return confirmed, tentative, rejected, ranking


# ---------------------------------------------------------------------------
# 3. SHAP Importance
# ---------------------------------------------------------------------------

def shap_importance(X, y, random_state=42):
    """Compute SHAP values for feature interpretation.

    Returns
    -------
    shap_scores : pd.Series — mean |SHAP| per feature, sorted descending.
    shap_values : np.ndarray — raw SHAP values (n_samples, n_features).
    explainer : shap.TreeExplainer — for further plotting.
    model : RandomForestClassifier — fitted model.
    """
    y_bin = (y == 1).astype(int)
    rf = RandomForestClassifier(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )
    rf.fit(X, y_bin)

    explainer = shap.TreeExplainer(rf)
    raw_shap = explainer.shap_values(X)

    # Handle different SHAP output formats
    if isinstance(raw_shap, list):
        shap_vals = np.abs(raw_shap[1])
    elif raw_shap.ndim == 3:
        shap_vals = np.abs(raw_shap[:, :, 1])
    else:
        shap_vals = np.abs(raw_shap)

    shap_scores = pd.Series(shap_vals.mean(axis=0), index=X.columns)
    return shap_scores.sort_values(ascending=False), shap_vals, explainer, rf


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_feature_selection(X, y, corr_threshold=0.9, random_state=42, verbose=True):
    """Run the 3-step feature selection pipeline.

    Steps:
      1. Correlation filter (remove redundant features)
      2. Boruta (validate remaining features statistically)

    SHAP is NOT run here — it should be computed after model training
    for proper interpretation.

    Parameters
    ----------
    X : pd.DataFrame — feature matrix.
    y : pd.Series — target labels (+1/-1).
    corr_threshold : float — correlation threshold for step 1.
    random_state : int
    verbose : bool

    Returns
    -------
    dict with keys:
        - corr_selected, corr_dropped, corr_matrix
        - target_corr: correlation of each feature with target
        - boruta_confirmed, boruta_tentative, boruta_rejected, boruta_ranking
        - selected: final list of features (Boruta confirmed from corr-filtered set)
    """
    results = {}

    # Step 1: Correlation filter
    if verbose:
        print("Step 1/2: Correlation filter...")
    corr_selected, corr_matrix, corr_dropped = correlation_filter(X, threshold=corr_threshold)
    target_corr = correlation_with_target(X, y)
    results["corr_selected"] = corr_selected
    results["corr_dropped"] = corr_dropped
    results["corr_matrix"] = corr_matrix
    results["target_corr"] = target_corr

    if verbose:
        print(f"  Dropped {len(corr_dropped)} redundant features: {corr_dropped}")
        print(f"  Remaining: {len(corr_selected)} features")

    # Step 2: Boruta on remaining features
    if verbose:
        print("Step 2/2: Boruta validation...")
    X_filtered = X[corr_selected]
    confirmed, tentative, rejected, ranking = boruta_selection(
        X_filtered, y, random_state=random_state
    )
    results["boruta_confirmed"] = confirmed
    results["boruta_tentative"] = tentative
    results["boruta_rejected"] = rejected
    results["boruta_ranking"] = ranking

    # Final selection: Boruta confirmed features
    results["selected"] = confirmed

    if verbose:
        print(f"  Confirmed: {confirmed}")
        if tentative:
            print(f"  Tentative: {tentative}")
        print(f"  Rejected: {rejected}")
        print(f"\nFinal selection: {len(confirmed)} features")

    return results
