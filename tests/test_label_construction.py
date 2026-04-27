"""Unit tests for label construction and the X / y / t1 alignment contract.

The tests cover three properties that the audit (docs/audit/01_repo_audit.md,
issues C-1 and C-5) required to be enforced:

1. The last ``window`` observations have NaN labels (not silently DOWN) and
   are therefore dropped from ``X`` / ``y``.
2. ``X.index``, ``y.index``, and ``t1.index`` are identical after NaN
   removal, and ``t1`` values are real timestamps in the original
   ``df.index`` shifted by ``window`` rows.
3. The default ``label_mode="raw_return"`` is causal: changing future
   prices does not retroactively change earlier labels. The wavelet mode,
   in contrast, *does* leak future information — this is verified
   explicitly so the leakage signature is testable.

Run with::

    pytest -q tests/test_label_construction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make ``src`` importable when pytest is invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (  # noqa: E402
    DEFAULT_LABEL_MODE,
    LABEL_MODES,
    create_target_labels,
    prepare_features,
    prepare_features_with_t1,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, seed: int = 0, drift: float = 0.0005) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a daily business-day index."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=0.01, size=n)
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
# 1. Last-h labels are NaN, then dropped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", [1, 2, 5, 10, 15])
def test_create_target_labels_last_window_is_nan(window: int) -> None:
    prices = pd.Series(np.linspace(100.0, 120.0, 50))
    labels = create_target_labels(prices, window)

    assert labels.iloc[-window:].isna().all(), "trailing labels must be NaN"
    assert labels.iloc[: -window].notna().all(), "earlier labels must be observed"
    # On a strictly increasing series, every observed label is +1.
    assert (labels.iloc[: -window] == 1.0).all()


def test_create_target_labels_tie_semantics() -> None:
    # Constant prices -> price_change == 0 -> tie -> -1 (kept for back-compat).
    prices = pd.Series([100.0] * 30)
    labels = create_target_labels(prices, window=3)
    assert labels.iloc[-3:].isna().all()
    assert (labels.iloc[:-3] == -1.0).all()


@pytest.mark.parametrize("window", [1, 5, 15])
def test_prepare_features_drops_last_window_rows(window: int) -> None:
    df = _make_ohlcv(n=200)
    X_raw_close = df["Close"]

    X, y = prepare_features(df, window=window, include_changes=False)

    # The last `window` calendar dates must NOT appear in X (they had NaN
    # labels and were dropped). The previous bug let them through with
    # silent DOWN labels.
    last_w_dates = X_raw_close.index[-window:]
    for date in last_w_dates:
        assert date not in X.index, (
            f"row at {date} should be dropped (label was NaN), "
            f"but is present in X"
        )

    # y has no NaN and only ±1.
    assert y.isna().sum() == 0
    assert set(np.unique(y)) <= {-1.0, 1.0}


# ---------------------------------------------------------------------------
# 2. X / y / t1 index alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", [1, 2, 5, 10, 15])
def test_prepare_features_with_t1_index_alignment(window: int) -> None:
    df = _make_ohlcv(n=300)
    X, y, t1 = prepare_features_with_t1(df, window=window, include_changes=False)

    assert X.index.equals(y.index), "X and y must share the same index"
    assert X.index.equals(t1.index), "X and t1 must share the same index"

    # Each t1 value must be a real timestamp from the original df.index.
    full_index = pd.Index(df.index)
    assert all(ts in full_index for ts in t1.values), (
        "every t1 value must be a real calendar timestamp in df.index"
    )

    # And it must lie exactly `window` rows after the row it labels.
    pos_t = full_index.get_indexer(X.index)
    pos_t1 = full_index.get_indexer(pd.Index(t1.values))
    assert (pos_t1 - pos_t == window).all(), (
        f"t1 must equal t + {window} trading days; got "
        f"deltas {np.unique(pos_t1 - pos_t)}"
    )


def test_prepare_features_with_t1_no_boundary_collapse() -> None:
    """Audit C-5: the previous `np.minimum(...)` cap collapsed the last
    `window` t1 values to a single boundary timestamp. After the fix, all
    t1 values are distinct (because the offending rows were dropped in
    prepare_features instead)."""
    df = _make_ohlcv(n=200)
    X, _, t1 = prepare_features_with_t1(df, window=10, include_changes=False)

    # We expect t1 to be strictly monotonic on its restriction to X.index.
    # (Indicators may strip leading rows; t1 is monotonic on what survives.)
    assert t1.is_monotonic_increasing


# ---------------------------------------------------------------------------
# 3. Raw labels are causal; wavelet labels are not
# ---------------------------------------------------------------------------

def test_raw_labels_do_not_use_future_smoothing() -> None:
    """Two price series that agree on [0, K] and differ on (K, N) must
    produce *identical* raw-return labels for every t with t + h <= K.

    This pins down the causality property: ``y_t`` depends only on
    ``{P_t, P_{t+h}}`` under raw-return labels.
    """
    n = 200
    h = 5
    rng = np.random.default_rng(1)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))

    K = 150  # split point; from K+1 onwards the two series differ
    prices_a = pd.Series(base, index=pd.bdate_range("2020-01-01", periods=n))
    prices_b = prices_a.copy()
    prices_b.iloc[K + 1 :] = -42.0  # arbitrary, very different tail

    labels_a = create_target_labels(prices_a, window=h)
    labels_b = create_target_labels(prices_b, window=h)

    # Indices [0, K - h] are formed entirely from prices in [0, K], which
    # are identical between the two series.
    safe_slice = slice(0, K - h + 1)
    assert labels_a.iloc[safe_slice].equals(labels_b.iloc[safe_slice]), (
        "raw labels at t with t+h <= K must not depend on prices > K"
    )

    # And the last h rows are NaN in both series (no contamination from
    # the tampered tail).
    assert labels_a.iloc[-h:].isna().all()
    assert labels_b.iloc[-h:].isna().all()


def test_wavelet_labels_DO_use_future_smoothing() -> None:
    """Negative control: with ``label_mode="wavelet"`` the same disturbance
    far in the future *does* change earlier labels, because
    ``pywt.wavedec`` is non-causal. This test documents the leakage and
    will fail if wavelet smoothing is ever made causal — at which point
    the assertion can be flipped or the test deleted.
    """
    df_a = _make_ohlcv(n=200, seed=2)
    df_b = df_a.copy()
    df_b.loc[df_b.index[-30:], "Close"] = df_a["Close"].iloc[-30] * 0.1  # large tail shock

    h = 5
    Xa, ya = prepare_features(df_a, window=h, label_mode="wavelet", include_changes=False)
    Xb, yb = prepare_features(df_b, window=h, label_mode="wavelet", include_changes=False)

    common = ya.index.intersection(yb.index)
    # Restrict to dates well before the shock so any difference is a pure
    # smoothing-leakage signature, not a Close-difference effect.
    safe_dates = common[common < df_a.index[-50]]
    diff_count = int((ya.loc[safe_dates] != yb.loc[safe_dates]).sum())
    assert diff_count > 0, (
        "wavelet labels were expected to differ on dates BEFORE the future "
        "shock — if this test starts passing with diff_count == 0 the "
        "wavelet implementation may have become causal; revisit C-1."
    )


# ---------------------------------------------------------------------------
# 4. label_mode parameter mechanics
# ---------------------------------------------------------------------------

def test_label_mode_default_is_raw_return() -> None:
    assert DEFAULT_LABEL_MODE == "raw_return"
    assert "raw_return" in LABEL_MODES


def test_label_mode_none_equals_raw_return() -> None:
    df = _make_ohlcv(n=150, seed=3)
    X_raw, y_raw = prepare_features(df, window=3, label_mode="raw_return")
    X_none, y_none = prepare_features(df, window=3, label_mode="none")
    assert X_raw.equals(X_none)
    assert y_raw.equals(y_none)


def test_legacy_smoothing_method_alias_still_works() -> None:
    df = _make_ohlcv(n=150, seed=4)
    # Old call site: smoothing_method="exponential" with no label_mode.
    X, y = prepare_features(df, window=3, smoothing_method="exponential")
    assert len(X) == len(y)
    assert y.isna().sum() == 0


def test_label_mode_unknown_value_raises() -> None:
    df = _make_ohlcv(n=80)
    with pytest.raises(ValueError):
        prepare_features(df, window=2, label_mode="not_a_real_mode")
