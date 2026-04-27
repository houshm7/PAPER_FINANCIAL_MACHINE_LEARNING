"""High-level analysis pipelines: K-Fold comparison, sector, window, portfolio."""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import f_oneway
import xgboost as xgb
from statsmodels.stats.multicomp import MultiComparison

from .config import CONFIG, MODEL_NAMES, STOCK_UNIVERSE
from .preprocessing import prepare_features_with_t1
from .validation import PurgedKFold
from .models import (
    create_models,
    calculate_metrics,
    _aggregate_fold_metrics,
    evaluate_with_standard_kfold,
    evaluate_with_purged_cv,
    evaluate_with_temporal_split,
)


# ---------------------------------------------------------------------------
# K-Fold comparison (standard vs purged)
# ---------------------------------------------------------------------------

def run_kfold_comparison(ticker, windows, stock_data, config=None, n_splits=5, pct_embargo=0.01,
                         feature_cols=None):
    """Compare Standard K-Fold vs Purged K-Fold for one stock across windows."""
    if config is None:
        config = CONFIG
    rows = []

    for window in windows:
        X, y, t1 = prepare_features_with_t1(stock_data[ticker], window, config,
                                              feature_cols=feature_cols)

        standard = evaluate_with_standard_kfold(
            X.copy(), y.copy(), create_models(config),
            n_splits=n_splits, random_state=config["random_state"],
        )
        purged = evaluate_with_purged_cv(
            X.copy(), y.copy(), t1, create_models(config),
            n_splits, pct_embargo, config,
        )

        for model_name in standard:
            for method, res in [("Standard K-Fold", standard), ("Purged K-Fold", purged)]:
                m = res[model_name]
                rows.append({
                    "Ticker": ticker,
                    "Window": window,
                    "Model": model_name,
                    "Method": method,
                    "Accuracy": m["accuracy"],
                    "Accuracy_Std": m.get("accuracy_std", 0),
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "Specificity": m["specificity"],
                    "F-Score": m["f_score"],
                    "AUC": m["auc"],
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detailed single-stock analysis
# ---------------------------------------------------------------------------

def run_detailed_single_stock_analysis(ticker, window, stock_data, config=None,
                                        feature_cols=None, n_splits=5, pct_embargo=0.01,
                                        hyperparams=None):
    """Purged K-Fold evaluation + temporal-split for visualizations.

    Returns Purged K-Fold metrics as primary results, plus temporal-split
    trained models for generating confusion matrices, ROC curves, etc.
    """
    if config is None:
        config = CONFIG

    X, y, t1 = prepare_features_with_t1(stock_data[ticker], window, config,
                                          feature_cols=feature_cols)
    models = create_models(config, hyperparams=hyperparams)

    # Primary evaluation: Purged K-Fold CV
    purged_results = evaluate_with_purged_cv(X, y, t1, models, n_splits, pct_embargo, config)

    # Temporal split: only for visualizations (trained models, predictions)
    _, trained_models, split = evaluate_with_temporal_split(X, y, t1, models, config)

    feature_names = list(X.columns)
    feature_importances = {}

    for model_name, mdata in trained_models.items():
        model = mdata["model"]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            imp = model.get_feature_importance()
        else:
            imp = np.zeros(len(feature_names))

        total = imp.sum()
        imp_pct = (imp / total) * 100 if total > 0 else np.zeros_like(imp)
        feature_importances[model_name] = {"raw": imp, "percentage": imp_pct}

    return {
        "ticker": ticker,
        "window": window,
        "results": purged_results,
        "trained_models": trained_models,
        "split": split,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
    }


# ---------------------------------------------------------------------------
# Single-stock multi-window
# ---------------------------------------------------------------------------

def run_single_stock_multiwindow_analysis(ticker, windows, stock_data, config=None,
                                           feature_cols=None, n_splits=5, pct_embargo=0.01,
                                           hyperparams=None):
    """Purged K-Fold evaluation across multiple windows for one stock."""
    if config is None:
        config = CONFIG
    rows = []

    for window in windows:
        X, y, t1 = prepare_features_with_t1(stock_data[ticker], window, config,
                                              feature_cols=feature_cols)
        models = create_models(config, hyperparams=hyperparams)
        eval_results = evaluate_with_purged_cv(X, y, t1, models, n_splits, pct_embargo, config)

        for model_name, metrics in eval_results.items():
            rows.append({
                "Ticker": ticker,
                "Window": window,
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Accuracy_Std": metrics.get("accuracy_std", 0),
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "F-Score": metrics["f_score"],
                "AUC": metrics["auc"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# All stocks — Purged K-Fold
# ---------------------------------------------------------------------------

def run_all_stocks_purged_cv(all_tickers, windows, stock_data, config=None, n_splits=5,
                              pct_embargo=0.01, feature_cols=None, hyperparams=None):
    """Purged K-Fold CV for every (ticker, window) combination."""
    if config is None:
        config = CONFIG
    rows = []
    t0 = time.time()

    for ticker in all_tickers:
        for window in windows:
            try:
                X, y, t1 = prepare_features_with_t1(stock_data[ticker], window, config,
                                                      feature_cols=feature_cols)
                models = create_models(config, hyperparams=hyperparams)
                pkf = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)

                for model_name, model in models.items():
                    fold_metrics = []
                    for train_idx, test_idx in pkf.split(X):
                        model_clone = clone(model)
                        y_train_bin = (y.iloc[train_idx] == 1).astype(int)
                        fit_kw = {}
                        if isinstance(model, (xgb.XGBClassifier, GradientBoostingClassifier)):
                            fit_kw["sample_weight"] = compute_sample_weight("balanced", y_train_bin)
                        model_clone.fit(X.iloc[train_idx], y_train_bin, **fit_kw)

                        y_pred = model_clone.predict(X.iloc[test_idx])
                        y_proba = (
                            model_clone.predict_proba(X.iloc[test_idx])[:, 1]
                            if hasattr(model_clone, "predict_proba")
                            else None
                        )
                        metrics = calculate_metrics(y.iloc[test_idx], y_pred * 2 - 1, y_proba)
                        fold_metrics.append(metrics)

                    agg = _aggregate_fold_metrics(fold_metrics)
                    rows.append({
                        "Ticker": ticker,
                        "Window": window,
                        "Model": model_name,
                        "Samples": len(X),
                        "Accuracy": agg["accuracy"],
                        "Accuracy_Std": agg["accuracy_std"],
                        "Precision": agg["precision"],
                        "Recall": agg["recall"],
                        "Specificity": agg["specificity"],
                        "F-Score": agg["f_score"],
                        "AUC": agg["auc"],
                    })
            except Exception:
                continue

    return pd.DataFrame(rows), time.time() - t0


# ---------------------------------------------------------------------------
# Sector-level statistics
# ---------------------------------------------------------------------------

def compute_sector_statistics(results_df, stock_universe=None):
    """Aggregate accuracy stats by sector and model."""
    if stock_universe is None:
        stock_universe = STOCK_UNIVERSE

    ticker_to_sector = {}
    for sector, info in stock_universe.items():
        for ticker in info["stocks"]:
            ticker_to_sector[ticker] = sector

    rws = results_df.copy()
    rws["Sector"] = rws["Ticker"].map(ticker_to_sector)

    rows = []
    for sector in stock_universe:
        sector_data = rws[rws["Sector"] == sector]
        for model in MODEL_NAMES:
            md = sector_data[sector_data["Model"] == model]
            if len(md) > 0:
                rows.append({
                    "Sector": sector,
                    "Model": model,
                    "Avg_Accuracy": md["Accuracy"].mean() * 100,
                    "Std": md["Accuracy"].std() * 100,
                    "Min": md["Accuracy"].min() * 100,
                    "Max": md["Accuracy"].max() * 100,
                    "Avg_AUC": md["AUC"].dropna().mean(),
                })

    return pd.DataFrame(rows), rws


# ---------------------------------------------------------------------------
# Window effect
# ---------------------------------------------------------------------------

def compute_window_statistics(results_df, windows=None):
    """Aggregate accuracy stats by window and model."""
    if windows is None:
        windows = CONFIG["windows"]

    rows = []
    for window in windows:
        wd = results_df[results_df["Window"] == window]
        for model in MODEL_NAMES:
            md = wd[wd["Model"] == model]
            if len(md) > 0:
                rows.append({
                    "Window": window,
                    "Model": model,
                    "Avg_Accuracy": md["Accuracy"].mean() * 100,
                    "Std": md["Accuracy"].std() * 100,
                    "Min": md["Accuracy"].min() * 100,
                    "Max": md["Accuracy"].max() * 100,
                    "Count": len(md),
                    "Avg_AUC": md["AUC"].dropna().mean(),
                })
    return pd.DataFrame(rows)


def compute_window_anova(results_df, windows=None):
    """ANOVA per model testing for a window effect on accuracy."""
    if windows is None:
        windows = CONFIG["windows"]

    results = {}
    for model in MODEL_NAMES:
        groups = []
        for window in windows:
            g = results_df[
                (results_df["Window"] == window) & (results_df["Model"] == model)
            ]["Accuracy"].values
            if len(g) > 0:
                groups.append(g)

        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            f_stat, p_value = f_oneway(*groups)
            results[model] = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
    return results


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def create_sector_portfolio(sector_stocks, stock_data, start_date=None, end_date=None):
    """Create an equal-weighted portfolio DataFrame from sector stocks."""
    available = [s for s in sector_stocks if s in stock_data]
    if not available:
        return None

    common_dates = None
    stock_dfs = {}
    for ticker in available:
        df = stock_data[ticker].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        common_dates = df.index if common_dates is None else common_dates.intersection(df.index)
        stock_dfs[ticker] = df

    if common_dates is None or len(common_dates) == 0:
        return None

    n = len(available)
    w = 1.0 / n
    port_ret = np.zeros(len(common_dates))
    port_vol = np.zeros(len(common_dates))
    port_high = np.zeros(len(common_dates))
    port_low = np.zeros(len(common_dates))
    port_open = np.zeros(len(common_dates))

    for ticker in available:
        df = stock_dfs[ticker].loc[common_dates]
        port_ret += w * df["Close"].pct_change().fillna(0).values
        port_vol += df["Volume"].values
        port_open += w * df["Open"].values
        port_high += w * df["High"].values
        port_low += w * df["Low"].values

    port_close = 100 * (1 + pd.Series(port_ret, index=common_dates)).cumprod()
    port_high = np.maximum(port_high, port_close.values)
    port_low = np.minimum(port_low, port_close.values)

    portfolio_df = pd.DataFrame(
        {"Open": port_open, "High": port_high, "Low": port_low,
         "Close": port_close.values, "Volume": port_vol},
        index=common_dates,
    )
    portfolio_df.index.name = "Date"
    portfolio_df = portfolio_df.replace([np.inf, -np.inf], np.nan).dropna()
    portfolio_df.loc[portfolio_df["Volume"] == 0, "Volume"] = 1
    return portfolio_df


def run_portfolio_analysis(stock_universe, stock_data, config=None, windows=None, n_splits=5,
                            pct_embargo=0.01, feature_cols=None, hyperparams=None):
    """Build sector portfolios, compute stats, and evaluate prediction models."""
    if config is None:
        config = CONFIG
    if windows is None:
        windows = config["windows"]

    sector_portfolios = {}
    for sector, info in stock_universe.items():
        pdf = create_sector_portfolio(info["stocks"], stock_data, config.get("start_date"), config.get("end_date"))
        if pdf is not None and len(pdf) > 100:
            sector_portfolios[sector] = pdf

    # Portfolio statistics
    stats_rows = []
    for sector, pdf in sector_portfolios.items():
        rets = pdf["Close"].pct_change().dropna()
        total_ret = (pdf["Close"].iloc[-1] / pdf["Close"].iloc[0] - 1) * 100
        ann_ret = ((pdf["Close"].iloc[-1] / pdf["Close"].iloc[0]) ** (252 / len(pdf)) - 1) * 100
        vol = rets.std() * np.sqrt(252) * 100
        sharpe = ann_ret / vol if vol > 0 else 0
        max_dd = ((pdf["Close"] / pdf["Close"].cummax() - 1).min()) * 100
        stats_rows.append({
            "Sector": sector, "Total_Return": total_ret, "Annualized_Return": ann_ret,
            "Volatility": vol, "Sharpe_Ratio": sharpe, "Max_Drawdown": max_dd,
        })
    portfolio_stats_df = pd.DataFrame(stats_rows)

    # Evaluate models on portfolios
    pred_rows = []
    for sector, pdf in sector_portfolios.items():
        for window in windows:
            try:
                X, y, t1 = prepare_features_with_t1(pdf, window, config,
                                                      feature_cols=feature_cols)
                res = evaluate_with_purged_cv(X, y, t1, create_models(config, hyperparams=hyperparams), n_splits, pct_embargo, config)
                for model_name, metrics in res.items():
                    pred_rows.append({
                        "Sector": sector, "Window": window, "Model": model_name,
                        "Accuracy": metrics["accuracy"],
                        "Accuracy_Std": metrics["accuracy_std"],
                        "Precision": metrics["precision"],
                        "Recall": metrics["recall"],
                        "Specificity": metrics["specificity"],
                        "F-Score": metrics["f_score"],
                        "AUC": metrics["auc"],
                    })
            except Exception:
                continue

    return sector_portfolios, pd.DataFrame(pred_rows), portfolio_stats_df


# ---------------------------------------------------------------------------
# Comparison & statistical tests
# ---------------------------------------------------------------------------

def compute_individual_vs_portfolio_comparison(results_df, portfolio_results_df, stock_universe=None):
    """Compare average individual stock accuracy per sector vs portfolio accuracy."""
    if stock_universe is None:
        stock_universe = STOCK_UNIVERSE

    ticker_to_sector = {}
    for sector, info in stock_universe.items():
        for ticker in info["stocks"]:
            ticker_to_sector[ticker] = sector

    rws = results_df.copy()
    if "Sector" not in rws.columns:
        rws["Sector"] = rws["Ticker"].map(ticker_to_sector)

    avg_ind = (
        rws.groupby(["Sector", "Model", "Window"])[
            ["Accuracy", "Accuracy_Std", "Precision", "Recall", "Specificity", "F-Score", "AUC"]
        ]
        .mean()
        .reset_index()
    )
    avg_ind["Type"] = "Individual Stocks (Avg)"

    port = portfolio_results_df.copy()
    port["Type"] = "Portfolio"

    cols = ["Sector", "Model", "Window", "Type", "Accuracy", "Accuracy_Std",
            "Precision", "Recall", "Specificity", "F-Score", "AUC"]
    return pd.concat([avg_ind[cols], port[cols]], ignore_index=True)


def compute_model_anova(results_df, portfolio_results_df=None):
    """ANOVA testing whether model accuracies differ (individual & portfolio)."""
    out = {}

    groups = [results_df[results_df["Model"] == m]["Accuracy"].values for m in MODEL_NAMES]
    if len(groups) == len(MODEL_NAMES) and all(len(g) > 0 for g in groups):
        f, p = f_oneway(*groups)
        out["individual"] = {"f_statistic": f, "p_value": p, "significant": p < 0.05}

    if portfolio_results_df is not None and not portfolio_results_df.empty:
        groups = [portfolio_results_df[portfolio_results_df["Model"] == m]["Accuracy"].values for m in MODEL_NAMES]
        if len(groups) == len(MODEL_NAMES) and all(len(g) > 0 for g in groups):
            f, p = f_oneway(*groups)
            out["portfolio"] = {"f_statistic": f, "p_value": p, "significant": p < 0.05}

    return out


def compute_tukey_hsd(results_df, perform_test=True):
    """Tukey HSD post-hoc test on model accuracies."""
    if not perform_test:
        return None
    try:
        mc = MultiComparison(results_df["Accuracy"], results_df["Model"])
        tukey = mc.tukeyhsd()
        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0],
        )
        means = results_df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
        return tukey, tukey_df, means
    except Exception:
        return None
