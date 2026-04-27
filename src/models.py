"""Model creation, metric computation, and evaluation pipelines."""

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from .config import CONFIG, MODEL_NAMES, TREE_MODEL_NAMES, BASELINE_NAMES
from .deep_learning import SklearnMLPClassifier, SklearnLSTMClassifier
from .validation import (
    PurgedKFold,
    get_standard_kfold_splits,
    temporal_train_test_split,
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_models(config=None, hyperparams=None):
    """Instantiate the five tree-based classifiers.

    Parameters
    ----------
    config : dict, optional
    hyperparams : dict, optional — per-model overrides.

    Returns
    -------
    dict[str, estimator]
    """
    if config is None:
        config = CONFIG
    if hyperparams is None:
        hyperparams = {}

    rs = config["random_state"]
    n_est = config["n_estimators"]
    n_jobs = config["n_jobs"]

    rf_p = hyperparams.get("Random Forest", {})
    xgb_p = hyperparams.get("XGBoost", {})
    gb_p = hyperparams.get("Gradient Boosting", {})
    lgb_p = hyperparams.get("LightGBM", {})
    cb_p = hyperparams.get("CatBoost", {})

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_p.get("n_estimators", n_est),
            criterion=rf_p.get("criterion", "gini"),
            max_features=rf_p.get("max_features", "sqrt"),
            max_depth=rf_p.get("max_depth", None),
            min_samples_split=rf_p.get("min_samples_split", 2),
            min_samples_leaf=rf_p.get("min_samples_leaf", 1),
            class_weight="balanced",
            random_state=rs,
            n_jobs=n_jobs,
            oob_score=True,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=xgb_p.get("n_estimators", n_est),
            max_depth=xgb_p.get("max_depth", 6),
            learning_rate=xgb_p.get("learning_rate", 0.3),
            subsample=xgb_p.get("subsample", 0.8),
            colsample_bytree=xgb_p.get("colsample_bytree", 0.8),
            min_child_weight=xgb_p.get("min_child_weight", 1),
            gamma=xgb_p.get("gamma", 0),
            objective="binary:logistic",
            random_state=rs,
            n_jobs=n_jobs,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=gb_p.get("n_estimators", n_est),
            max_depth=gb_p.get("max_depth", 3),
            learning_rate=gb_p.get("learning_rate", 0.1),
            subsample=gb_p.get("subsample", 0.8),
            min_samples_split=gb_p.get("min_samples_split", 2),
            min_samples_leaf=gb_p.get("min_samples_leaf", 1),
            random_state=rs,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=lgb_p.get("n_estimators", n_est),
            max_depth=lgb_p.get("max_depth", 7),
            learning_rate=lgb_p.get("learning_rate", 0.05),
            num_leaves=lgb_p.get("num_leaves", 31),
            subsample=lgb_p.get("subsample", 0.8),
            colsample_bytree=lgb_p.get("colsample_bytree", 0.8),
            min_child_samples=lgb_p.get("min_child_samples", 20),
            is_unbalance=True,
            random_state=rs,
            n_jobs=n_jobs,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=cb_p.get("iterations", n_est),
            depth=cb_p.get("depth", 6),
            learning_rate=cb_p.get("learning_rate", 0.03),
            subsample=cb_p.get("subsample", 0.8),
            min_data_in_leaf=cb_p.get("min_data_in_leaf", 1),
            auto_class_weights="Balanced",
            random_state=rs,
            verbose=False,
            thread_count=n_jobs,
        ),
    }


def create_baseline_models(config=None):
    """Instantiate baseline models for comparison.

    Returns
    -------
    dict[str, estimator]
        - Dummy (Most Frequent): always predicts the majority class.
        - Dummy (Stratified): random predictions respecting class proportions.
        - Logistic Regression: simple linear baseline.
    """
    if config is None:
        config = CONFIG
    rs = config["random_state"]

    return {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent", random_state=rs),
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=rs),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=rs, n_jobs=config["n_jobs"]),
    }


def create_dl_models(config=None, hyperparams=None, input_dim=10):
    """Instantiate deep learning classifiers.

    Parameters
    ----------
    config : dict, optional
    hyperparams : dict, optional — per-model overrides.
    input_dim : int — number of input features.

    Returns
    -------
    dict[str, estimator]
    """
    if config is None:
        config = CONFIG
    if hyperparams is None:
        hyperparams = {}

    rs = config["random_state"]
    mlp_p = hyperparams.get("MLP", {})
    lstm_p = hyperparams.get("LSTM", {})

    return {
        "MLP": SklearnMLPClassifier(
            input_dim=input_dim,
            hidden_dim=mlp_p.get("hidden_dim", 64),
            n_layers=mlp_p.get("n_layers", 2),
            dropout=mlp_p.get("dropout", 0.3),
            lr=mlp_p.get("lr", 1e-3),
            max_epochs=mlp_p.get("max_epochs", 100),
            batch_size=mlp_p.get("batch_size", 32),
            random_state=rs,
        ),
        "LSTM": SklearnLSTMClassifier(
            input_dim=input_dim,
            hidden_dim=lstm_p.get("hidden_dim", 32),
            n_lstm_layers=lstm_p.get("n_lstm_layers", 1),
            seq_len=lstm_p.get("seq_len", 5),
            dropout=lstm_p.get("dropout", 0.2),
            lr=lstm_p.get("lr", 1e-3),
            max_epochs=lstm_p.get("max_epochs", 100),
            batch_size=lstm_p.get("batch_size", 32),
            random_state=rs,
        ),
    }


def create_stacking_model(t1, config=None, hyperparams=None,
                           n_splits=5, pct_embargo=0.01):
    """Create stacking ensemble with Purged K-Fold internal CV.

    Base learners: 4 tree-based (RF, XGBoost, GB, LightGBM).
    CatBoost excluded — lacks __sklearn_tags__ compatibility.
    Meta-learner: Logistic Regression.

    Parameters
    ----------
    t1 : pd.Series — label end times (required for PurgedKFold).
    """
    from sklearn.ensemble import StackingClassifier

    if config is None:
        config = CONFIG
    if hyperparams is None:
        hyperparams = {}

    base = create_models(config, hyperparams)
    # CatBoost lacks __sklearn_tags__ — incompatible with StackingClassifier
    base.pop("CatBoost", None)

    estimators = list(base.items())

    return {
        "Stacking": StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                max_iter=1000, random_state=config["random_state"]
            ),
            cv=PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo),
            stack_method="predict_proba",
        )
    }


def create_all_models(config=None, hyperparams=None, input_dim=10):
    """Instantiate baselines + tree-based + DL classifiers together.

    Returns
    -------
    dict[str, estimator]
    """
    models = create_baseline_models(config)
    models.update(create_models(config, hyperparams))
    models.update(create_dl_models(config, hyperparams, input_dim))
    return models


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Compute accuracy, precision, recall, specificity, F1, AUC, and confusion counts.

    Labels are expected as +1/-1; they are converted to 0/1 internally.
    """
    y_true_bin = (y_true == 1).astype(int)
    y_pred_bin = (y_pred == 1).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f_score": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    if y_proba is not None:
        try:
            if hasattr(y_proba, "ndim"):
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    y_proba_pos = y_proba[:, 1]
                elif y_proba.ndim == 2 and y_proba.shape[1] == 1:
                    y_proba_pos = y_proba.ravel()
                else:
                    y_proba_pos = y_proba
            else:
                y_proba_pos = y_proba
            metrics["auc"] = roc_auc_score(y_true_bin, y_proba_pos)
        except Exception:
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_fold_metrics(fold_metrics):
    """Average metric dicts across CV folds."""
    auc_values = [m["auc"] for m in fold_metrics if m["auc"] is not None]
    return {
        "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
        "accuracy_std": np.std([m["accuracy"] for m in fold_metrics]),
        "precision": np.mean([m["precision"] for m in fold_metrics]),
        "recall": np.mean([m["recall"] for m in fold_metrics]),
        "specificity": np.mean([m["specificity"] for m in fold_metrics]),
        "f_score": np.mean([m["f_score"] for m in fold_metrics]),
        "auc": np.mean(auc_values) if auc_values else None,
    }


def _needs_scaling(model):
    """Check if model is a DL wrapper that benefits from feature scaling."""
    return isinstance(model, (SklearnMLPClassifier, SklearnLSTMClassifier))


def _needs_sample_weight(model):
    """Check if model needs external sample_weight for class balancing.

    XGBoost and GradientBoosting don't have a native 'class_weight' param,
    so we pass balanced sample weights via fit().
    """
    return isinstance(model, (xgb.XGBClassifier, GradientBoostingClassifier))


def _train_predict(model, X_train, y_train, X_test):
    """Clone, fit, predict (binary labels + probabilities).

    Applies StandardScaler automatically for DL models (MLP, LSTM).
    Applies balanced sample_weight for XGBoost and GradientBoosting.
    """
    model_clone = clone(model)
    y_train_bin = (y_train == 1).astype(int)

    # Scale features for DL models
    scaler = None
    if _needs_scaling(model_clone):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Balanced sample weights for models without native class_weight
    fit_kwargs = {}
    if _needs_sample_weight(model_clone):
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train_bin)

    model_clone.fit(X_train, y_train_bin, **fit_kwargs)

    y_pred = model_clone.predict(X_test)
    y_proba = (
        model_clone.predict_proba(X_test)[:, 1]
        if hasattr(model_clone, "predict_proba")
        else None
    )
    return model_clone, y_pred, y_proba


# ---------------------------------------------------------------------------
# Evaluation pipelines
# ---------------------------------------------------------------------------

def evaluate_with_standard_kfold(X, y, models, n_splits=5, random_state=42):
    """Evaluate models with standard (shuffled) K-Fold — causes data leakage."""
    results = {}
    for model_name, model in models.items():
        fold_metrics = []
        splits = get_standard_kfold_splits(X, y, n_splits=n_splits, random_state=random_state)

        for train_idx, test_idx in splits:
            _, y_pred, y_proba = _train_predict(
                model, X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx]
            )
            metrics = calculate_metrics(y.iloc[test_idx], y_pred * 2 - 1, y_proba)
            fold_metrics.append(metrics)

        results[model_name] = _aggregate_fold_metrics(fold_metrics)
    return results


def evaluate_with_purged_cv(X, y, t1, models, n_splits=5, pct_embargo=0.01, config=None):
    """Evaluate models with Purged K-Fold CV (correct for finance)."""
    if config is None:
        config = CONFIG
    pkf = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    results = {}

    for model_name, model in models.items():
        fold_metrics = []
        for train_idx, test_idx in pkf.split(X):
            _, y_pred, y_proba = _train_predict(
                model, X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx]
            )
            metrics = calculate_metrics(y.iloc[test_idx], y_pred * 2 - 1, y_proba)
            fold_metrics.append(metrics)

        avg = {}
        for key in fold_metrics[0]:
            if key not in ("tp", "tn", "fp", "fn"):
                values = [m[key] for m in fold_metrics if m[key] is not None]
                avg[key] = np.mean(values) if values else None
                avg[f"{key}_std"] = np.std(values) if values else None
        avg["model"] = model_name
        avg["method"] = "Purged K-Fold CV"
        results[model_name] = avg

    return results


def evaluate_with_temporal_split(X, y, t1, models, config=None):
    """Evaluate models with a chronological train/test split."""
    if config is None:
        config = CONFIG
    split = temporal_train_test_split(X, y, t1, test_size=config["test_size"])

    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]

    results = {}
    trained_models = {}

    for model_name, model in models.items():
        model_clone, y_pred, y_proba = _train_predict(model, X_train, y_train, X_test)

        y_test_bin = (y_test == 1).astype(int)
        metrics = calculate_metrics(y_test, y_pred * 2 - 1, y_proba)
        metrics["model"] = model_name
        metrics["method"] = "Temporal Split"

        results[model_name] = metrics
        trained_models[model_name] = {
            "model": model_clone,
            "y_test": y_test_bin,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    return results, trained_models, split
