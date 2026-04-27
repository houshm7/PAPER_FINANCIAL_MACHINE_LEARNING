"""Preprocessing pipeline: smoothing methods, label creation, feature preparation.

Smoothing methods available:
  - "exponential" : Basak et al. (2019) — exponential smoothing (α=0.095)
  - "wavelet"     : Wavelet denoising (DWT with soft thresholding)
  - "savgol"      : Savitzky-Golay filter (local polynomial fitting)
  - "none"        : No smoothing (raw prices)

Key improvement over Basak et al.:
  - Indicators are computed on RAW data (they have their own internal smoothing)
  - Smoothing is applied ONLY to the Close price for label creation
  - This avoids double-smoothing and preserves indicator sensitivity
"""

import numpy as np
import pandas as pd
import pywt
from scipy.signal import savgol_filter as _savgol_filter

from .config import CONFIG, FEATURE_COLS, ORIGINAL_FEATURE_COLS, EXTENDED_FEATURE_COLS, CHANGE_FEATURE_COLS
from .indicators import calculate_all_indicators

# Available smoothing methods
SMOOTHING_METHODS = ["exponential", "wavelet", "savgol", "none"]


# ---------------------------------------------------------------------------
# Smoothing methods
# ---------------------------------------------------------------------------

def exponential_smoothing(series, alpha=0.095):
    """Apply exponential smoothing to a time series.

    Parameters
    ----------
    series : pd.Series
    alpha : float — smoothing factor (default from Basak et al., 2019).
        Lower α = more smoothing. α=0.095 is aggressive.

    Returns
    -------
    pd.Series
    """
    smoothed = series.copy()
    smoothed.iloc[0] = series.iloc[0]
    for t in range(1, len(series)):
        smoothed.iloc[t] = alpha * series.iloc[t] + (1 - alpha) * smoothed.iloc[t - 1]
    return smoothed


def wavelet_denoising(series, wavelet="db4", level=None, mode="soft"):
    """Wavelet denoising using Discrete Wavelet Transform (DWT).

    Decomposes the signal into frequency components, applies soft/hard
    thresholding to high-frequency coefficients (noise), and reconstructs.

    Advantages over exponential smoothing:
      - Preserves sharp transitions and local features
      - Adapts to local signal characteristics
      - Based on multi-resolution analysis

    Parameters
    ----------
    series : pd.Series — input time series.
    wavelet : str — wavelet family (default "db4" = Daubechies-4).
    level : int, optional — decomposition level. If None, auto-determined.
    mode : str — "soft" (default) or "hard" thresholding.

    Returns
    -------
    pd.Series — denoised time series.

    References
    ----------
    Wang et al. (2025) — "Wavelet Denoising and Double-Layer Feature Selection
    for Stock Trend Prediction", Computational Economics.
    """
    values = series.values.astype(float)

    if level is None:
        level = min(pywt.dwt_max_level(len(values), wavelet), 4)

    coeffs = pywt.wavedec(values, wavelet, level=level)

    # Universal threshold (VisuShrink): σ * sqrt(2 * log(n))
    # Estimate noise σ from finest detail coefficients (MAD estimator)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(values)))

    # Apply thresholding to detail coefficients (keep approximation intact)
    denoised_coeffs = [coeffs[0]]  # approximation — untouched
    for detail in coeffs[1:]:
        if mode == "soft":
            denoised = pywt.threshold(detail, threshold, mode="soft")
        else:
            denoised = pywt.threshold(detail, threshold, mode="hard")
        denoised_coeffs.append(denoised)

    reconstructed = pywt.waverec(denoised_coeffs, wavelet)

    # waverec may produce array slightly longer than input due to padding
    reconstructed = reconstructed[:len(values)]

    return pd.Series(reconstructed, index=series.index, name=series.name)


def savgol_smoothing(series, window_length=21, polyorder=3):
    """Savitzky-Golay filter — local polynomial fitting.

    Fits a polynomial of degree `polyorder` to successive windows of
    `window_length` data points, evaluating the polynomial at the center.

    Advantages:
      - Preserves peaks and slopes better than moving average
      - Purely local operation (no look-ahead if applied causally)

    Parameters
    ----------
    series : pd.Series
    window_length : int — must be odd. Window size for local fitting.
    polyorder : int — polynomial degree (must be < window_length).

    Returns
    -------
    pd.Series

    References
    ----------
    Springer (2025) — "An efficient framework for accurate stock market price
    prediction using bidirectional GRU and Savitzky-Golay filter".
    """
    if window_length % 2 == 0:
        window_length += 1  # Must be odd

    smoothed = _savgol_filter(series.values, window_length, polyorder)
    return pd.Series(smoothed, index=series.index, name=series.name)


def apply_smoothing(series, method="exponential", config=None):
    """Apply the specified smoothing method to a price series.

    Parameters
    ----------
    series : pd.Series — raw prices.
    method : str — one of "exponential", "wavelet", "savgol", "none".
    config : dict, optional

    Returns
    -------
    pd.Series — smoothed prices (or original if method="none").
    """
    if config is None:
        config = CONFIG

    if method == "none":
        return series.copy()
    elif method == "exponential":
        return exponential_smoothing(series, alpha=config["alpha"])
    elif method == "wavelet":
        return wavelet_denoising(
            series,
            wavelet=config.get("wavelet", "db4"),
            level=config.get("wavelet_level", None),
        )
    elif method == "savgol":
        return savgol_smoothing(
            series,
            window_length=config.get("savgol_window", 21),
            polyorder=config.get("savgol_polyorder", 3),
        )
    else:
        raise ValueError(
            f"Unknown smoothing method: '{method}'. "
            f"Choose from {SMOOTHING_METHODS}"
        )


def compute_noise_reduction(original, smoothed):
    """Compute noise reduction percentage between original and smoothed series.

    Returns
    -------
    float — noise reduction in percent.
    """
    diff_original = original.diff().dropna()
    diff_smoothed = smoothed.diff().dropna()
    return (1 - diff_smoothed.std() / diff_original.std()) * 100


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------

def create_target_labels(prices, window):
    """Create binary labels: +1 if price goes up over *window* days, else -1.

    Parameters
    ----------
    prices : pd.Series
    window : int

    Returns
    -------
    pd.Series
    """
    future_price = prices.shift(-window)
    price_change = future_price - prices
    labels = np.where(price_change > 0, 1, -1)
    return pd.Series(labels, index=prices.index)


# ---------------------------------------------------------------------------
# Change (Δ) features
# ---------------------------------------------------------------------------

def add_change_features(df_indicators, base_cols=None):
    """Add 1-day change (Δ) features for each indicator.

    For each indicator X, computes X_CHG = X(t) - X(t-1).
    This captures the *dynamics* of each indicator — whether it is rising,
    falling, or flat — which tree-based models cannot infer from a single
    snapshot.

    Parameters
    ----------
    df_indicators : pd.DataFrame — DataFrame with indicator columns.
    base_cols : list[str], optional — indicator columns to diff.
        Defaults to EXTENDED_FEATURE_COLS.

    Returns
    -------
    pd.DataFrame — copy of input with ``{col}_CHG`` columns appended.
    """
    if base_cols is None:
        base_cols = EXTENDED_FEATURE_COLS
    result = df_indicators.copy()
    for col in base_cols:
        if col in result.columns:
            result[f"{col}_CHG"] = result[col].diff()
    return result


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df, window, config=None, feature_cols=None, extended=True,
                     smoothing_method="wavelet", include_changes=False):
    """Full preprocessing pipeline (improved).

    Key difference from Basak et al. (2019):
      - Indicators are computed on RAW data (no double-smoothing)
      - Smoothing is applied ONLY to Close price for label creation

    Steps:
      1. Technical indicators on RAW data
      2. (Optional) Add 1-day change features for each indicator
      3. Smoothing on Close price (for labels only)
      4. Binary target labels from smoothed Close
      5. Drop NaN rows

    Parameters
    ----------
    df : pd.DataFrame — raw OHLCV data.
    window : int — trading window in days.
    config : dict, optional
    feature_cols : list[str], optional — override feature columns.
    extended : bool — if True, compute all 14 indicators; if False, only original 6.
    smoothing_method : str — "exponential", "wavelet", "savgol", or "none".
    include_changes : bool — if True, append Δ features (1-day change per indicator).

    Returns
    -------
    X : pd.DataFrame — feature matrix.
    y : pd.Series    — target labels (+1 / -1).
    """
    if config is None:
        config = CONFIG

    base_cols = EXTENDED_FEATURE_COLS if extended else ORIGINAL_FEATURE_COLS

    # Step 1: Indicators on RAW data (no smoothing applied to features)
    df_indicators = calculate_all_indicators(df, config, extended=extended)

    # Step 2: Add change features if requested or if feature_cols contains _CHG columns
    needs_changes = include_changes or (
        feature_cols is not None and any(c.endswith("_CHG") for c in feature_cols)
    )
    if needs_changes:
        df_indicators = add_change_features(df_indicators, base_cols=base_cols)

    # Determine which columns to use
    if feature_cols is not None:
        use_cols = [c for c in feature_cols if c in df_indicators.columns]
    elif include_changes:
        change_cols = [f"{c}_CHG" for c in base_cols if f"{c}_CHG" in df_indicators.columns]
        use_cols = base_cols + change_cols
    else:
        use_cols = base_cols

    # Step 3: Smoothing on Close ONLY for label creation
    smoothed_close = apply_smoothing(df["Close"], method=smoothing_method, config=config)

    # Step 4: Labels from smoothed Close
    labels = create_target_labels(smoothed_close, window)

    # Step 5: Combine and drop NaN
    df_features = df_indicators[use_cols].copy()
    df_features["target"] = labels
    df_features = df_features.dropna()

    X = df_features[use_cols]
    y = df_features["target"]
    return X, y


def prepare_features_basak(df, window, config=None, feature_cols=None, extended=False):
    """Original Basak et al. (2019) pipeline — for comparison only.

    Smoothing is applied BEFORE indicators (double-smoothing).
    Uses original 6 features by default.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if config is None:
        config = CONFIG
    if feature_cols is None:
        feature_cols = ORIGINAL_FEATURE_COLS if not extended else EXTENDED_FEATURE_COLS

    # Basak: smooth FIRST, then compute indicators on smoothed data
    df_smoothed = df.copy()
    df_smoothed["Close"] = exponential_smoothing(df["Close"], alpha=config["alpha"])

    df_indicators = calculate_all_indicators(df_smoothed, config, extended=extended)
    labels = create_target_labels(df_smoothed["Close"], window)

    df_features = df_indicators[feature_cols].copy()
    df_features["target"] = labels
    df_features = df_features.dropna()

    X = df_features[feature_cols]
    y = df_features["target"]
    return X, y


def prepare_features_with_t1(df, window, config=None, feature_cols=None, extended=True,
                              smoothing_method="wavelet", include_changes=False):
    """Preprocessing pipeline extended with t1 series for Purged K-Fold CV.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    t1 : pd.Series — maps each observation index to its label end time.
    """
    if config is None:
        config = CONFIG

    X, y = prepare_features(
        df, window, config,
        feature_cols=feature_cols, extended=extended,
        smoothing_method=smoothing_method,
        include_changes=include_changes,
    )

    end_positions = np.minimum(np.arange(len(X.index)) + window, len(X.index) - 1)
    t1 = pd.Series(X.index[end_positions], index=X.index)

    return X, y, t1
