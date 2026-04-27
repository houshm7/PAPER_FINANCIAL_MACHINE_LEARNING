"""Unit tests for the economic-backtest module.

Each test pins one slice of the backtest contract:

1. All-up predictions on a strictly rising series → return ≈ buy-and-hold.
2. All-down predictions on a strictly rising series → realised loss.
3. Perfect predictions → realised gain ≫ buy-and-hold.
4. Cost sensitivity: net return at c=0 vs c=10 bps differs by exactly
   ``n_trades × c``.
5. Threshold strategy reduces trade count vs sign strategy.
6. Equity curve never NaN / never inf.
7. Non-overlapping h-day execution: n_trades = floor((n_predictions
   that survive boundary) / h).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import realize_strategy, sweep_backtests  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rising_prices():
    """Strictly increasing daily price series, 252 trading days."""
    idx = pd.bdate_range("2024-01-01", periods=252)
    return pd.Series(np.linspace(100.0, 200.0, 252), index=idx)


@pytest.fixture
def rising_predictions(rising_prices):
    """One prediction per trading day. y_pred = +1 (always up)."""
    return pd.DataFrame({
        "date": rising_prices.index,
        "y_pred": np.ones(len(rising_prices), dtype=int),
        "y_proba": np.full(len(rising_prices), 0.7),
    })


# ---------------------------------------------------------------------------
# 1. Long-only on rising series ≈ buy-and-hold
# ---------------------------------------------------------------------------

def test_all_long_on_rising_matches_buy_and_hold(rising_prices, rising_predictions):
    res = realize_strategy(
        rising_predictions, rising_prices,
        horizon=1, cost_bps=0.0, strategy="sign",
    )
    # On a strictly rising price series with all-long h=1 daily round trips,
    # gross compounded return equals buy-and-hold over the same span.
    np.testing.assert_allclose(
        res.gross_total_return,
        res.buy_and_hold_total_return,
        rtol=1e-9,
    )
    assert res.n_trades > 0
    assert res.hit_rate == 1.0
    assert res.n_long == res.n_trades
    assert res.n_short == 0


# ---------------------------------------------------------------------------
# 2. All-short on rising series → realised loss
# ---------------------------------------------------------------------------

def test_all_short_on_rising_loses_money(rising_prices):
    preds = pd.DataFrame({
        "date": rising_prices.index,
        "y_pred": -np.ones(len(rising_prices), dtype=int),
        "y_proba": np.full(len(rising_prices), 0.3),
    })
    res = realize_strategy(preds, rising_prices, horizon=1, cost_bps=0.0)
    assert res.gross_total_return < 0
    assert res.hit_rate == 0.0
    assert res.n_short == res.n_trades
    assert res.n_long == 0


# ---------------------------------------------------------------------------
# 3. Perfect predictions outperform buy-and-hold
# ---------------------------------------------------------------------------

def test_perfect_predictions_dominate_buy_and_hold():
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.bdate_range("2024-01-01", periods=n)
    rets = rng.normal(0, 0.02, size=n)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx)
    # Perfect oracle: y_pred[t] = sign(price[t+1] - price[t]).
    next_prices = prices.shift(-1)
    y_pred = np.array(np.sign(next_prices - prices).fillna(0).astype(int).values, copy=True)
    y_pred[y_pred == 0] = 1  # avoid zero positions in degenerate ties
    preds = pd.DataFrame({
        "date": idx,
        "y_pred": y_pred,
        "y_proba": (y_pred > 0).astype(float),
    })
    res = realize_strategy(preds, prices, horizon=1, cost_bps=0.0)
    # An h=1 oracle's gross return should be non-negative and at least
    # as large as buy-and-hold (it always rides the up days and avoids
    # the down days).
    assert res.gross_total_return > res.buy_and_hold_total_return
    assert res.hit_rate >= 0.99  # essentially perfect


# ---------------------------------------------------------------------------
# 4. Cost sensitivity is exact
# ---------------------------------------------------------------------------

def test_cost_subtracts_per_trade(rising_prices, rising_predictions):
    res0  = realize_strategy(rising_predictions, rising_prices,
                             horizon=1, cost_bps=0.0)
    res10 = realize_strategy(rising_predictions, rising_prices,
                             horizon=1, cost_bps=10.0)
    assert res0.n_trades == res10.n_trades
    # Each trade pays cost_bps/1e4. Per-trade NET return diff should be
    # exactly that.
    per_trade_diff = (res0.trades["net_return"] - res10.trades["net_return"]).mean()
    np.testing.assert_allclose(per_trade_diff, 10.0 / 1e4, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Threshold strategy is more selective
# ---------------------------------------------------------------------------

def test_threshold_strategy_reduces_trade_count(rising_prices):
    # Mix of high-conviction and low-conviction signals.
    n = len(rising_prices)
    half = n // 2
    probas = np.concatenate([np.full(half, 0.55), np.full(n - half, 0.95)])
    preds = pd.DataFrame({
        "date": rising_prices.index,
        "y_pred": np.ones(n, dtype=int),
        "y_proba": probas,
    })
    res_sign = realize_strategy(preds, rising_prices, horizon=1,
                                cost_bps=0.0, strategy="sign")
    res_thr  = realize_strategy(preds, rising_prices, horizon=1,
                                cost_bps=0.0, strategy="threshold",
                                threshold=0.10)
    # Threshold 0.10 means only |proba - 0.5| > 0.10, i.e. proba > 0.6
    # or proba < 0.4. Half the signals (the 0.55s) are filtered out.
    assert res_thr.n_trades < res_sign.n_trades
    assert res_thr.n_trades == n - half - 1  # last row has no exit at h=1


# ---------------------------------------------------------------------------
# 6. Equity curve sanity
# ---------------------------------------------------------------------------

def test_equity_curve_finite(rising_prices, rising_predictions):
    res = realize_strategy(rising_predictions, rising_prices,
                           horizon=1, cost_bps=10.0)
    eq = res.equity_curve
    assert eq["gross_equity"].notna().all()
    assert eq["net_equity"].notna().all()
    assert np.isfinite(eq["gross_equity"]).all()
    assert np.isfinite(eq["net_equity"]).all()
    assert (eq["gross_equity"] > 0).all()


# ---------------------------------------------------------------------------
# 7. Non-overlapping h-day execution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h", [1, 2, 5, 10])
def test_n_trades_matches_non_overlapping_count(rising_prices, h):
    n = len(rising_prices)
    preds = pd.DataFrame({
        "date": rising_prices.index,
        "y_pred": np.ones(n, dtype=int),
    })
    res = realize_strategy(preds, rising_prices, horizon=h, cost_bps=0.0)
    # Picks every h-th row (0, h, 2h, ...) and drops trades whose exit
    # would overrun. Number kept = ceil(n/h) minus those whose +h
    # overruns. The last entry kept is the largest k*h with k*h+h < n.
    expected = sum(1 for k in range(0, n, h) if k + h < n)
    assert res.n_trades == expected


# ---------------------------------------------------------------------------
# 8. Sweep API smoke
# ---------------------------------------------------------------------------

def test_sweep_smoke(rising_prices):
    n = len(rising_prices)
    preds = pd.DataFrame({
        "date": list(rising_prices.index) * 2,
        "model": ["A"] * n + ["B"] * n,
        "y_pred": [1] * n + [-1] * n,
        "y_proba": [0.7] * n + [0.3] * n,
    })
    metrics, equity = sweep_backtests(
        preds, rising_prices,
        horizon=1,
        cost_bps_grid=(0.0, 10.0),
        strategies=("sign",),
        ticker="TEST",
    )
    # 2 models x 2 costs x 1 strategy = 4 rows
    assert len(metrics) == 4
    assert {"A", "B"} == set(metrics["model"])
    assert {0.0, 10.0} == set(metrics["cost_bps"])
    # On rising series: A (long) gross > 0, B (short) gross < 0
    assert metrics.loc[metrics["model"] == "A", "gross_total_return"].iloc[0] > 0
    assert metrics.loc[metrics["model"] == "B", "gross_total_return"].iloc[0] < 0
