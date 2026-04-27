"""Hyperparameter tuning with Optuna using Purged K-Fold CV."""

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from .config import CONFIG, MODEL_NAMES
from .deep_learning import SklearnMLPClassifier, SklearnLSTMClassifier
from .validation import PurgedKFold
from .models import calculate_metrics

# Silence Optuna logs by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def _rf_objective(trial, X, y, t1, n_splits, pct_embargo, config):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    model = RandomForestClassifier(
        **params, class_weight="balanced",
        random_state=config["random_state"], n_jobs=config["n_jobs"],
    )
    return _cv_score(model, X, y, t1, n_splits, pct_embargo)


def _xgb_objective(trial, X, y, t1, n_splits, pct_embargo, config):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    model = xgb.XGBClassifier(
        **params, objective="binary:logistic",
        random_state=config["random_state"], n_jobs=config["n_jobs"],
    )
    return _cv_score(model, X, y, t1, n_splits, pct_embargo)


def _gb_objective(trial, X, y, t1, n_splits, pct_embargo, config):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    model = GradientBoostingClassifier(
        **params, random_state=config["random_state"],
    )
    return _cv_score(model, X, y, t1, n_splits, pct_embargo)


def _lgb_objective(trial, X, y, t1, n_splits, pct_embargo, config):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }
    model = lgb.LGBMClassifier(
        **params, is_unbalance=True,
        random_state=config["random_state"],
        n_jobs=config["n_jobs"], verbose=-1,
    )
    return _cv_score(model, X, y, t1, n_splits, pct_embargo)


def _catboost_objective(trial, X, y, t1, n_splits, pct_embargo, config):
    params = {
        "iterations": trial.suggest_int("iterations", 50, 500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    model = CatBoostClassifier(
        **params, auto_class_weights="Balanced",
        random_state=config["random_state"],
        verbose=False, thread_count=config["n_jobs"],
    )
    return _cv_score(model, X, y, t1, n_splits, pct_embargo)


def _mlp_objective(trial, X, y, t1, n_splits, pct_embargo, config):
    params = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "max_epochs": trial.suggest_categorical("max_epochs", [50, 100, 150]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    model = SklearnMLPClassifier(
        input_dim=X.shape[1], **params,
        random_state=config["random_state"],
    )
    return _cv_score(model, X, y, t1, n_splits, pct_embargo)


def _lstm_objective(trial, X, y, t1, n_splits, pct_embargo, config):
    params = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
        "n_lstm_layers": trial.suggest_int("n_lstm_layers", 1, 2),
        "seq_len": trial.suggest_int("seq_len", 3, 10),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "max_epochs": trial.suggest_categorical("max_epochs", [50, 100]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    model = SklearnLSTMClassifier(
        input_dim=X.shape[1], **params,
        random_state=config["random_state"],
    )
    return _cv_score(model, X, y, t1, n_splits, pct_embargo)


_OBJECTIVE_MAP = {
    "Random Forest": _rf_objective,
    "XGBoost": _xgb_objective,
    "Gradient Boosting": _gb_objective,
    "LightGBM": _lgb_objective,
    "CatBoost": _catboost_objective,
    "MLP": _mlp_objective,
    "LSTM": _lstm_objective,
}


# ---------------------------------------------------------------------------
# Cross-validation scorer (Purged K-Fold)
# ---------------------------------------------------------------------------

def _cv_score(model, X, y, t1, n_splits, pct_embargo):
    """Return mean accuracy across Purged K-Fold splits."""
    from .deep_learning import SklearnMLPClassifier, SklearnLSTMClassifier

    pkf = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    needs_scale = isinstance(model, (SklearnMLPClassifier, SklearnLSTMClassifier))
    needs_sw = isinstance(model, (xgb.XGBClassifier, GradientBoostingClassifier))
    accuracies = []

    for train_idx, test_idx in pkf.split(X):
        model_clone = clone(model)
        y_train_bin = (y.iloc[train_idx] == 1).astype(int)

        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        if needs_scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        fit_kw = {}
        if needs_sw:
            fit_kw["sample_weight"] = compute_sample_weight("balanced", y_train_bin)

        model_clone.fit(X_tr, y_train_bin, **fit_kw)

        y_pred = model_clone.predict(X_te)
        metrics = calculate_metrics(y.iloc[test_idx], y_pred * 2 - 1)
        accuracies.append(metrics["accuracy"])

    return np.mean(accuracies)


# ---------------------------------------------------------------------------
# Main tuning function
# ---------------------------------------------------------------------------

def tune_model(model_name, X, y, t1, n_trials=50, n_splits=5, pct_embargo=0.01, config=None):
    """Tune a single model's hyperparameters with Optuna + Purged K-Fold.

    Parameters
    ----------
    model_name : str
        One of MODEL_NAMES.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels (+1 / -1).
    t1 : pd.Series
        Label end times for purging.
    n_trials : int
        Number of Optuna trials (more = better but slower).
    n_splits, pct_embargo : int, float
        Purged K-Fold parameters.
    config : dict, optional

    Returns
    -------
    dict with keys: best_params, best_accuracy, study
    """
    if config is None:
        config = CONFIG

    if model_name not in _OBJECTIVE_MAP:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(_OBJECTIVE_MAP)}")

    objective_fn = _OBJECTIVE_MAP[model_name]

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_fn(trial, X, y, t1, n_splits, pct_embargo, config),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return {
        "best_params": study.best_params,
        "best_accuracy": study.best_value,
        "study": study,
    }


def tune_all_models(X, y, t1, n_trials=50, n_splits=5, pct_embargo=0.01, config=None, model_names=None):
    """Tune all (or selected) tree-based models.

    Parameters
    ----------
    model_names : list[str], optional
        Models to tune. Defaults to all MODEL_NAMES.

    Returns
    -------
    dict[str, dict] — {model_name: {best_params, best_accuracy, study}}
    """
    if config is None:
        config = CONFIG
    if model_names is None:
        model_names = MODEL_NAMES

    results = {}
    for name in model_names:
        print(f"\nTuning {name} ({n_trials} trials)...")
        results[name] = tune_model(name, X, y, t1, n_trials, n_splits, pct_embargo, config)
        print(f"  Best accuracy: {results[name]['best_accuracy']:.4f}")
        print(f"  Best params: {results[name]['best_params']}")

    return results


def build_tuned_hyperparams(tuning_results):
    """Convert Optuna results to the hyperparams dict expected by create_models().

    Parameters
    ----------
    tuning_results : dict
        Output of tune_all_models().

    Returns
    -------
    dict — ready to pass to create_models(config, hyperparams=...).
    """
    hyperparams = {}
    for model_name, result in tuning_results.items():
        hyperparams[model_name] = result["best_params"]
    return hyperparams
