"""Reusable plotting functions for the stock prediction project.

All paper-facing figures use a unified publication style (setup_paper_style)
and are saved via _save_fig() with consistent DPI and tight bounding boxes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

from .config import MODEL_NAMES, MODEL_COLORS, MODEL_MARKERS

# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

# Publication palette — muted, print-friendly
PAPER_PALETTE = {
    "Random Forest": "#d62728",
    "XGBoost": "#1f77b4",
    "Gradient Boosting": "#2ca02c",
    "LightGBM": "#9467bd",
    "CatBoost": "#ff7f0e",
}

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

EXPORT_DPI = 300


def setup_paper_style():
    """Apply publication-quality Matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PAPER_RC)


def setup_style():
    """Legacy alias kept for backward compatibility."""
    setup_paper_style()


def _save_fig(fig, filename, close=True):
    """Save figure to the figures/ directory and optionally close it."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    print(f"  Saved: {path}")


def _model_color(name):
    return PAPER_PALETTE.get(name, MODEL_COLORS.get(name, "#333333"))


# ---------------------------------------------------------------------------
# EDA plots (not saved to paper figures)
# ---------------------------------------------------------------------------

def plot_normalized_prices(stock_data, representative_stocks):
    """Normalized price evolution (base = 100) for representative stocks."""
    setup_paper_style()
    n = len(representative_stocks)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.array(axes).flatten()
    for idx, (sector, ticker) in enumerate(representative_stocks.items()):
        if ticker not in stock_data:
            continue
        df = stock_data[ticker]
        norm = (df["Close"] / df["Close"].iloc[0]) * 100
        axes[idx].plot(df.index, norm, linewidth=1.2, color="steelblue")
        axes[idx].axhline(y=100, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
        axes[idx].set_title(f"{sector}: {ticker}")
        axes[idx].set_ylabel("Normalized price")
        axes[idx].tick_params(axis="x", rotation=45)
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Normalized Price Evolution by Sector", fontsize=12, y=1.00)
    plt.tight_layout()
    plt.show()


def plot_return_distributions(stock_data, representative_stocks):
    """Histogram of daily returns with stats box."""
    setup_paper_style()
    n = len(representative_stocks)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.array(axes).flatten()
    for idx, (sector, ticker) in enumerate(representative_stocks.items()):
        if ticker not in stock_data:
            continue
        rets = stock_data[ticker]["Close"].pct_change().dropna() * 100
        ax = axes[idx]
        ax.hist(rets, bins=50, alpha=0.7, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(rets.mean(), color="red", linestyle="--", linewidth=1.2, label=f"Mean: {rets.mean():.3f}%")
        ax.set_title(f"{ticker} ({sector})")
        ax.set_xlabel("Daily return (%)")
        ax.legend(fontsize=7)
        stats_text = f"$\\sigma$={rets.std():.2f}%\nSkew={rets.skew():.2f}\nKurt={rets.kurtosis():.2f}"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Daily Return Distributions by Sector", fontsize=12, y=1.00)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr_matrix, title="Correlation Matrix", save_as=None):
    """Generic heatmap for a correlation matrix."""
    setup_paper_style()
    n = len(corr_matrix)
    size = max(5, min(8, 0.55 * n))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.75},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title(title, fontsize=11, pad=12)
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    if save_as:
        _save_fig(fig, save_as, close=False)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Indicator plots
# ---------------------------------------------------------------------------

def plot_indicator_with_price(dates, prices, indicator, indicator_name,
                              overbought=None, oversold=None, ticker=""):
    """Two-panel chart: price on top, indicator on bottom."""
    setup_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [1.4, 1]})
    ax1.plot(dates, prices, linewidth=1.2, color="black", label="Close")
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"{ticker} — {indicator_name}", fontsize=11)
    ax1.legend(fontsize=7)
    ax2.plot(dates, indicator, linewidth=1.2, color="#1f77b4", label=indicator_name)
    if overbought is not None:
        ax2.axhline(y=overbought, color="red", linestyle="--", alpha=0.6, linewidth=0.8)
    if oversold is not None:
        ax2.axhline(y=oversold, color="green", linestyle="--", alpha=0.6, linewidth=0.8)
    ax2.set_ylabel(indicator_name)
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=7)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Paper figures — model comparison plots
# ---------------------------------------------------------------------------

def plot_kfold_comparison(comparison_df, windows, analysis_ticker):
    """Six-panel chart comparing Standard vs Purged K-Fold across metrics.

    Saves: kfold_comparison.png
    """
    setup_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F-Score", "AUC"]
    method_styles = {
        "Standard K-Fold": {"color": "#d62728", "marker": "o", "ls": "-"},
        "Purged K-Fold":   {"color": "#1f77b4", "marker": "s", "ls": "-"},
    }
    x = np.arange(len(windows))
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        # Individual model points (faint)
        for mn in MODEL_NAMES:
            offset = (MODEL_NAMES.index(mn) - 2) * 0.06
            for method, sty in method_styles.items():
                vals = comparison_df[
                    (comparison_df["Model"] == mn) & (comparison_df["Method"] == method)
                ][metric].values
                if len(vals):
                    ax.plot(x + offset, vals, marker=MODEL_MARKERS[mn], linestyle="None",
                            markersize=4, color=sty["color"], alpha=0.25)
        # Average lines
        for method, sty in method_styles.items():
            avg = comparison_df[comparison_df["Method"] == method].groupby("Window")[metric].mean()
            ax.plot(x, avg.values, marker=sty["marker"], linestyle=sty["ls"],
                    color=sty["color"], linewidth=1.8, markersize=6, label=f"{method} (avg)")
        ax.set_xlabel("Window (days)")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(windows)
        if idx == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"Standard vs Purged K-Fold — {analysis_ticker}",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _save_fig(fig, "kfold_comparison.png", close=False)
    plt.show()
    plt.close(fig)


def plot_feature_importance(feature_names, feature_importances, ticker, window):
    """Bar charts of feature importance for each model.

    Saves: boruta_feature_importance.png (only the global importance panel)
    """
    setup_paper_style()
    n_models = min(len(MODEL_NAMES), 5)
    fig, axes = plt.subplots(1, n_models, figsize=(3.2 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
    cmap = plt.cm.viridis
    for idx, model_name in enumerate(MODEL_NAMES[:n_models]):
        ax = axes[idx]
        imp_pct = feature_importances[model_name]["percentage"]
        order = np.argsort(imp_pct)[::-1]
        colors = cmap(np.linspace(0.25, 0.75, len(feature_names)))
        ax.barh([feature_names[i] for i in order], imp_pct[order],
                color=colors, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Importance (%)")
        ax.set_title(model_name, fontsize=10)
        for j, v in enumerate(imp_pct[order]):
            if v > 1:
                ax.text(v + 0.3, j, f"{v:.1f}", va="center", fontsize=7)
    fig.suptitle(f"Feature Importance — {ticker} (window={window}d)", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_roc_curves(trained_models):
    """Overlay ROC curves for all models.

    Saves: roc_curves_comparison.png
    """
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model_name, mdata in trained_models.items():
        y_test = mdata["y_test"]
        y_proba = mdata["y_proba"]
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, linewidth=1.5, color=_model_color(model_name),
                label=f"{model_name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Tuned Models", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    _save_fig(fig, "roc_curves_comparison.png", close=False)
    plt.show()
    plt.close(fig)


def plot_confusion_matrices(trained_models, ticker, window):
    """Confusion matrices for each model.

    Saves: confusion_matrices.png
    """
    setup_paper_style()
    n = min(len(MODEL_NAMES), 5)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for idx, model_name in enumerate(MODEL_NAMES[:n]):
        ax = axes[idx]
        mdata = trained_models[model_name]
        cm = confusion_matrix(mdata["y_test"], mdata["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Down", "Up"], yticklabels=["Down", "Up"],
                    cbar=False, annot_kws={"size": 10})
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("Predicted")
        if idx == 0:
            ax.set_ylabel("Actual")
        else:
            ax.set_ylabel("")
    fig.suptitle(f"Confusion Matrices — {ticker} ({window}d)", fontsize=11, y=1.02)
    plt.tight_layout()
    _save_fig(fig, "confusion_matrices.png", close=False)
    plt.show()
    plt.close(fig)


def plot_accuracy_vs_window(single_stock_df, ticker):
    """Line chart of accuracy across trading windows for each model.

    Saves: window_accuracy_heatmap.png (as the main window-effect figure)
    """
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in MODEL_NAMES:
        md = single_stock_df[single_stock_df["Model"] == model]
        ax.plot(md["Window"], md["Accuracy"] * 100, marker=MODEL_MARKERS[model],
                linewidth=1.5, markersize=6, color=_model_color(model), label=model)
    ax.set_xlabel("Forecasting horizon (days)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy vs Horizon — {ticker}", fontsize=11)
    ax.legend(fontsize=7)
    ax.axhline(50, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.tight_layout()
    _save_fig(fig, "window_accuracy_heatmap.png", close=False)
    plt.show()
    plt.close(fig)


def plot_sector_bar_chart(sector_df, sectors):
    """Grouped bar chart of average accuracy by sector and model.

    Saves: sector_analysis.png
    """
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sectors))
    width = 0.15
    for i, model in enumerate(MODEL_NAMES):
        accs = []
        for sector in sectors:
            row = sector_df[(sector_df["Sector"] == sector) & (sector_df["Model"] == model)]
            accs.append(row["Avg_Accuracy"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, accs, width, label=model, color=_model_color(model),
               edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Sector")
    ax.set_ylabel("Average accuracy (%)")
    ax.set_title("Model Performance by Sector", fontsize=11)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(sectors)
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    _save_fig(fig, "sector_analysis.png", close=False)
    plt.show()
    plt.close(fig)


def plot_sector_heatmap(sector_df, sectors):
    """Heatmap of accuracy: sectors x models.

    Saves: sector_heatmap_results.png
    """
    setup_paper_style()
    matrix = []
    for sector in sectors:
        row = []
        for model in MODEL_NAMES:
            sub = sector_df[(sector_df["Sector"] == sector) & (sector_df["Model"] == model)]
            row.append(sub["Avg_Accuracy"].values[0] if len(sub) > 0 else 0)
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=MODEL_NAMES, yticklabels=sectors, ax=ax,
                linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Average Accuracy (%) by Sector and Model", fontsize=11, pad=10)
    ax.tick_params(axis="both", labelsize=9)
    plt.tight_layout()
    _save_fig(fig, "sector_heatmap_results.png", close=False)
    plt.show()
    plt.close(fig)


def plot_window_effect(window_df, windows):
    """Line plot + box plot for window effect.

    (Exploratory — not a primary paper figure)
    """
    setup_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1 = axes[0]
    for model in MODEL_NAMES:
        md = window_df[window_df["Model"] == model].sort_values("Window")
        ax1.plot(md["Window"], md["Avg_Accuracy"], marker=MODEL_MARKERS[model],
                 linewidth=1.5, markersize=6, color=_model_color(model), label=model)
        ax1.fill_between(md["Window"],
                         md["Avg_Accuracy"] - md["Std"],
                         md["Avg_Accuracy"] + md["Std"],
                         alpha=0.10, color=_model_color(model))
    ax1.set_xlabel("Window (days)")
    ax1.set_ylabel("Avg accuracy (%)")
    ax1.set_title("Accuracy vs Window ($\\pm 1\\sigma$)")
    ax1.legend(fontsize=7)
    ax2 = axes[1]
    box_data = [window_df[window_df["Window"] == w]["Avg_Accuracy"].values for w in windows]
    bp = ax2.boxplot(box_data, tick_labels=[str(w) for w in windows], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4CAF50")
        patch.set_alpha(0.5)
    ax2.set_title("Accuracy Distribution by Window")
    ax2.set_xlabel("Window (days)")
    ax2.set_ylabel("Avg accuracy (%)")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_window_heatmap(window_df, windows):
    """Heatmap of accuracy: windows x models.

    (Supplementary)
    """
    setup_paper_style()
    matrix = []
    for window in windows:
        row = []
        for model in MODEL_NAMES:
            sub = window_df[(window_df["Window"] == window) & (window_df["Model"] == model)]
            row.append(sub["Avg_Accuracy"].values[0] if len(sub) > 0 else 0)
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=MODEL_NAMES, yticklabels=[f"{w}d" for w in windows], ax=ax,
                linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Average Accuracy (%) by Window and Model", fontsize=11, pad=10)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_portfolio_evolution(sector_portfolios):
    """Line chart of portfolio cumulative value per sector.

    Saves: portfolio_evolution.png
    """
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sector, pdf in sector_portfolios.items():
        ax.plot(pdf.index, pdf["Close"], linewidth=1.5, label=sector, alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value (base = 100)")
    ax.set_title("Equal-Weighted Sector Portfolio Evolution", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_fig(fig, "portfolio_evolution.png", close=False)
    plt.show()
    plt.close(fig)


def plot_individual_vs_portfolio(comparison_df, sectors):
    """Side-by-side accuracy comparison for individual stocks vs portfolios.

    Saves: individual_vs_portfolio.png
    """
    setup_paper_style()
    n_models = min(len(MODEL_NAMES), 5)
    fig, axes = plt.subplots(1, n_models, figsize=(3.2 * n_models, 3.8),
                              sharey=True)
    if n_models == 1:
        axes = [axes]
    for idx, model in enumerate(MODEL_NAMES[:n_models]):
        ax = axes[idx]
        md = comparison_df[comparison_df["Model"] == model]
        for t in md["Type"].unique():
            sub = md[md["Type"] == t].groupby("Window")["Accuracy"].mean()
            marker = "o" if "Individual" in t else "s"
            ls = "--" if "Individual" in t else "-"
            ax.plot(sub.index, sub.values * 100, marker=marker, linewidth=1.3,
                    linestyle=ls, label=t, markersize=5)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Window (days)")
        if idx == 0:
            ax.set_ylabel("Accuracy (%)")
            ax.legend(fontsize=6)
    fig.suptitle("Individual vs Portfolio Accuracy", fontsize=11, y=1.02)
    plt.tight_layout()
    _save_fig(fig, "individual_vs_portfolio.png", close=False)
    plt.show()
    plt.close(fig)
