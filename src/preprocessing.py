"""Preprocessing pipeline: smoothing methods, label creation, feature preparation.

Label modes
-----------
The main research pipeline now defaults to ``label_mode="raw_return"``: the
binary direction label is the sign of the *raw* h-day forward price change
``P_{t+h} - P_t``. Raw returns are causal — the label at time ``t`` depends
only on ``{P_t, P_{t+h}}`` and never on prices after ``t+h``.

Smoothed-label modes are retained as robustness checks only:

- ``"exponential"`` — Basak et al. (2019) exponential smoothing (α=0.095).
  Causal but introduces lag; not equivalent to the raw label.
- ``"wavelet"`` — DWT denoising (Daubechies-4 by default). **Non-causal**:
  the implementation here uses a global ``pywt.wavedec`` over the full
  series, so smoothed prices at time ``t`` depend on prices ``> t``.
  Reported only as a robustness check; not for headline numbers.
- ``"savgol"`` — Savitzky-Golay centred polynomial filter. **Non-causal**.
- ``"none"`` — alias for ``"raw_return"`` (no smoothing).

For backward compatibility every API still accepts the legacy
``smoothing_method=`` parameter; when both are passed, ``label_mode`` wins.

Key improvement over Basak et al. (2019)
----------------------------------------
- Indicators are computed on RAW data (they have their own internal smoothing).
- Smoothing — when used at all — is applied ONLY to the Close price used
  for label construction; features are never smoothed.
"""

import warnings
import numpy as np
import pandas as pd
import pywt
from scipy.signal import savgol_filter as _savgol_filter

from .config import CONFIG, FEATURE_COLS, ORIGINAL_FEATURE_COLS, EXTENDED_FEATURE_COLS, CHANGE_FEATURE_COLS
from .indicators import calculate_all_indicators

# Available smoothing methods (legacy "smoothing_method" parameter values)
SMOOTHING_METHODS = ["exponential", "wavelet", "savgol", "none"]

# Canonical label-construction modes. ``raw_return`` and ``none`` are
# equivalent; both skip smoothing and use raw forward returns for the sign.
LABEL_MODES = ["raw_return", "wavelet", "exponential", "savgol", "none"]
DEFAULT_LABEL_MODE = "raw_return"


def _resolve_label_mode(label_mode, smoothing_method):
    """Map (label_mode, smoothing_method) to a single smoothing method string.

    Precedence:
      - explicit ``smoothing_method`` (legacy callers) wins.
      - otherwise ``label_mode`` is used; ``"raw_return"`` and ``"none"``
        both resolve to ``"none"``.
    """
    if smoothing_method is not None:
        if smoothing_method not in SMOOTHING_METHODS:
            raise ValueError(
                f"Unknown smoothing_method {smoothing_method!r}. "
                f"Choose from {SMOOTHING_METHODS}."
            )
        return smoothing_method
    if label_mode is None:
        label_mode = DEFAULT_LABEL_MODE
    if label_mode not in LABEL_MODES:
        raise ValueError(
            f"Unknown label_mode {label_mode!r}. Choose from {LABEL_MODES}."
        )
    if label_mode in ("raw_return", "none"):
        return "none"
    return label_mode


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
    """Create binary direction labels with explicit NaN at the boundary.

    For each row ``t``, the label is

    - ``+1`` if ``prices[t+window] > prices[t]``,
    - ``-1`` if ``prices[t+window] <= prices[t]`` (ties → DOWN, kept for
      backward compatibility),
    - ``NaN`` for the last ``window`` rows where ``prices[t+window]`` is
      not observed.

    The previous implementation used ``np.where(price_change > 0, 1, -1)``
    which silently mapped the ``NaN`` future-price comparison to ``-1``,
    inflating the DOWN class and contaminating the panel with degenerate
    labels. Callers should now drop the trailing ``NaN`` rows (this is
    handled automatically by :func:`prepare_features`).

    Parameters
    ----------
    prices : pd.Series
        Reference price series (e.g. raw close, or a smoothed variant
        depending on ``label_mode``). Indexed by the trading-day calendar.
    window : int
        Forecasting horizon in rows.

    Returns
    -------
    pd.Series
        Float64 series indexed identically to ``prices``; values in
        ``{+1, -1, NaN}``.
    """
    future_price = prices.shift(-window)
    price_change = future_price - prices

    labels = pd.Series(np.nan, index=prices.index, dtype="float64")
    valid = price_change.notna()
    # Tie semantics preserved: price_change == 0 → -1.
    labels.loc[valid & (price_change > 0)] = 1.0
    labels.loc[valid & (price_change <= 0)] = -1.0
    return labels


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
                     label_mode=DEFAULT_LABEL_MODE, include_changes=False,
                     smoothing_method=None):
    """Full preprocessing pipeline.

    Pipeline
    --------
    1. Compute technical indicators on RAW OHLCV (no double-smoothing).
    2. Optionally append Δ features for each indicator.
    3. Resolve ``label_mode`` (default ``"raw_return"``); apply the chosen
       smoothing operator to ``df["Close"]`` *only* for label construction.
    4. Build binary direction labels via :func:`create_target_labels`. The
       last ``window`` rows are ``NaN`` because ``P_{t+h}`` is unobserved.
    5. Drop NaN rows. Indicator warm-up rows and the trailing
       label-NaN rows are removed in this single step, so ``X`` and ``y``
       share the same index by construction.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data indexed by trading day.
    window : int
        Forecasting horizon in trading days (``h``).
    config : dict, optional
    feature_cols : list[str], optional
        Override the feature columns. If unset, ``EXTENDED_FEATURE_COLS``
        (or ``ORIGINAL_FEATURE_COLS`` when ``extended=False``) is used,
        plus Δ features when ``include_changes=True``.
    extended : bool
        If ``True``, compute all 14 indicators; if ``False``, only the
        six original Basak et al. (2019) indicators.
    label_mode : str
        One of ``LABEL_MODES``. Default is ``"raw_return"`` (the only
        causal label construction). Smoothed alternatives are retained
        for robustness.
    include_changes : bool
        If ``True``, append Δ features (1-day change per indicator).
    smoothing_method : str, optional
        Deprecated alias for ``label_mode``. When provided, takes
        precedence over ``label_mode`` for backward compatibility.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels (+1 / -1) sharing ``X``'s index.
    """
    if config is None:
        config = CONFIG

    effective_smoothing = _resolve_label_mode(label_mode, smoothing_method)

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

    # Step 3: Smoothing on Close ONLY for label creation. ``"none"`` is the
    # raw-return path: the smoothing helper returns a copy of the input.
    smoothed_close = apply_smoothing(df["Close"], method=effective_smoothing, config=config)

    # Step 4: Labels (last ``window`` rows are NaN by construction)
    labels = create_target_labels(smoothed_close, window)

    # Step 5: Combine and drop NaN. Both indicator warm-up rows and the
    # trailing label-NaN rows disappear here, so X.index == y.index.
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
                              label_mode=DEFAULT_LABEL_MODE, include_changes=False,
                              smoothing_method=None):
    """Preprocessing pipeline extended with the ``t1`` series for Purged K-Fold CV.

    ``t1[i]`` is the calendar timestamp at which the label for observation
    ``X.iloc[i]`` is *fully formed* — that is, the timestamp of
    ``P_{t+h}``. It is taken from the **original** ``df.index`` so it
    remains correct even though :func:`prepare_features` drops the last
    ``window`` rows when forming ``X`` and ``y``.

    The previous implementation built ``t1`` against the cleaned
    ``X.index`` and capped overruns with ``np.minimum`` (audit C-5),
    which collapsed the last ``window`` ``t1`` values to a single
    boundary timestamp. With the NaN-aware label fix that capping is no
    longer needed: every retained row has a real ``t+h`` timestamp in
    ``df.index``.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    t1 : pd.Series
        Indexed identically to ``X``; values are timestamps drawn from
        the original ``df.index``.
    """
    if config is None:
        config = CONFIG

    X, y = prepare_features(
        df, window, config,
        feature_cols=feature_cols, extended=extended,
        label_mode=label_mode,
        include_changes=include_changes,
        smoothing_method=smoothing_method,
    )

    full_index = pd.Index(df.index)
    pos = full_index.get_indexer(X.index)
    if (pos < 0).any():
        # Should not happen — X.index is always a subset of df.index.
        raise RuntimeError(
            "prepare_features_with_t1: X.index is not a subset of df.index; "
            "cannot resolve t1 timestamps."
        )
    end_pos = pos + window
    if end_pos.max() >= len(full_index):
        # Should not happen — the trailing NaN labels were dropped in
        # prepare_features, so every retained row has t+h in df.index.
        raise RuntimeError(
            "prepare_features_with_t1: t1 lookup overran df.index. "
            "Check that create_target_labels still NaNs the last `window` rows."
        )
    t1 = pd.Series(full_index[end_pos], index=X.index)
    return X, y, t1
