# Forecasting Stock Market Direction with Tree-Based Classifiers

Evidence from Purged Cross-Validation and SHAP Interpretability.

**Authors:** Houssem Majed, Aymen El Ghoul, Hanaa Talbi, Rania Hentati Kaffel
*Université Paris 1 Panthéon-Sorbonne — École d'Économie de la Sorbonne*

## Abstract

We benchmark five tree-based classifiers — Random Forest, XGBoost, Gradient Boosting, LightGBM, and CatBoost — on a panel of 25 U.S. equities from five sectors over 2020–2024. From raw OHLCV data we compute 14 technical indicators and their first differences; a two-stage selection procedure (correlation filter + Boruta) retains 13 validated predictors. We show that standard K-fold cross-validation inflates accuracy by 11 to 33 percentage points relative to Purged K-Fold with embargo, with the bias growing monotonically with the label horizon. Under leakage-free evaluation, the best tuned model attains 73% directional accuracy at the one-day horizon. SHAP analysis identifies the Relative Strength Index as the dominant predictor across all architectures.

**Keywords:** Stock market prediction, tree-based classifiers, Purged K-Fold Cross-Validation, information leakage, SHAP, technical indicators.
**JEL:** G14, G17, C45, C53.

## Repository Structure

```
.
├── src/                  # Python source modules
│   ├── data.py             # Data loading (yfinance)
│   ├── indicators.py       # Technical indicator computation
│   ├── preprocessing.py    # Cleaning, scaling, smoothing
│   ├── feature_selection.py# Correlation filter + Boruta
│   ├── models.py           # Tree-based classifier wrappers
│   ├── tuning.py           # Optuna hyperparameter search
│   ├── validation.py       # Purged K-Fold CV with embargo
│   ├── analysis.py         # SHAP, performance analysis
│   ├── deep_learning.py    # Neural baseline (skorch / PyTorch)
│   ├── visualization.py    # Figure generation
│   └── config.py           # Experiment configuration
├── final_notebook/       # End-to-end Jupyter notebook
├── final_paper/          # LaTeX source of the paper (main.tex, references.bib)
├── academic_report/      # Companion academic report
├── journal_qf_template/  # Quantitative Finance journal LaTeX template
├── reference_paper/      # Reference literature
├── figures/              # Generated figures used in the paper
├── results/              # Experiment outputs (e.g. purged_cv_results.csv)
├── pyproject.toml        # Project dependencies (uv)
└── uv.lock               # Locked dependency versions
```

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
# Clone
git clone https://github.com/houshm7/PAPER_FINANCIAL_MACHINE_LEARNING.git
cd PAPER_FINANCIAL_MACHINE_LEARNING

# Install dependencies (creates .venv automatically)
uv sync

# Activate the environment
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

## GPU Support (optional)

The pipeline runs on CPU by default and uses CUDA opportunistically when a
GPU is available. GPU usage is gated by the `use_gpu` flag in
`src/config.py` (default `True`); set it to `False` to force CPU.

| Model              | GPU path                                        | CPU fallback |
|--------------------|-------------------------------------------------|--------------|
| XGBoost            | `device="cuda"`, `tree_method="hist"`           | yes          |
| CatBoost           | `task_type="GPU"`, `bootstrap_type="Bernoulli"` | yes          |
| LightGBM           | `device_type="gpu"` (only if the build supports it; auto-probed) | yes |
| MLP / LSTM (torch) | `device="cuda"`                                 | yes          |
| Random Forest, sklearn Gradient Boosting | n/a (sklearn has no CUDA path)        | CPU only     |

Run the diagnostic to see what your environment supports:

```bash
python scripts/check_gpu.py
```

The script trains a tiny model on each backend with the GPU configuration
that `src.gpu` would select and prints a per-backend OK/-- line. Stock pip
LightGBM wheels are CPU-only; the helper detects this on first call and
keeps LightGBM on CPU instead of failing mid-training.

## Reproducing the Results

The full pipeline is documented in [`final_notebook/`](final_notebook/). Open the notebook in Jupyter / VS Code and run cells sequentially:

```bash
jupyter lab final_notebook/
```

The notebook covers:

1. **Data acquisition** — 25 U.S. equities from 5 sectors (2020–2024) via `yfinance`.
2. **Feature engineering** — 14 technical indicators + first differences.
3. **Feature selection** — correlation filter followed by Boruta (13 predictors retained).
4. **Validation comparison** — standard K-Fold vs. Purged K-Fold with embargo.
5. **Model training & tuning** — five tree-based classifiers tuned with Optuna.
6. **Interpretability** — SHAP analysis across all architectures.
7. **Portfolio-level evaluation** — aggregation across stocks and sectors.

Generated figures are written to `figures/` and tabular results to `results/`.

## Key Findings

- Standard K-fold CV inflates accuracy by **11–33 percentage points** vs. Purged K-Fold; the bias grows with label horizon.
- Under leakage-free evaluation, the best tuned model reaches **73% directional accuracy** at the 1-day horizon (~10 pp above a persistence baseline).
- Portfolio-level prediction improves accuracy by **up to 13 points** over individual stocks.
- **RSI** is the dominant predictor across all five architectures, with a SHAP profile consistent with short-term momentum continuation.

## Paper

The compiled paper is built from [`final_paper/main.tex`](final_paper/main.tex) using the Quantitative Finance journal template in [`journal_qf_template/`](journal_qf_template/).

## Citation

If you use this work, please cite:

```bibtex
@unpublished{majed2025forecasting,
  title  = {Forecasting Stock Market Direction with Tree-Based Classifiers:
            Evidence from Purged Cross-Validation and SHAP Interpretability},
  author = {Majed, Houssem and El Ghoul, Aymen and Talbi, Hanaa
            and Hentati Kaffel, Rania},
  year   = {2025},
  note   = {Université Paris 1 Panthéon-Sorbonne}
}
```
