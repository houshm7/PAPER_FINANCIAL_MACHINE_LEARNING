"""Economic backtest of leakage-safe directional predictions.

Translates the out-of-fold predictions produced by
``src.pipeline.run_nested_purged_cv`` into a realised P&L stream and
computes the metrics that a Quantitative Finance referee actually
cares about: net annualised return, Sharpe, max drawdown, hit rate,
turnover — net of round-trip transaction costs.

Audit cross-reference: critical issue C-19
("No realised P&L, Sharpe, or transaction-cost backtest").

Strategy spec
-------------
For each prediction ``(t, model, y_pred ∈ {−1, +1}, y_proba)`` at
horizon ``h``:

- **Sign strategy** (default): position ``p_t = y_pred``. Long if the
  model predicts up, short if it predicts down.
- **Threshold strategy**: position is taken only when
  ``|y_proba − 0.5| > δ`` (high-conviction trades only).

Trades are executed *non-overlapping at horizon h*: enter at the
close of day ``t``, exit at the close of day ``t + h``. For ``h = 1``
this is daily round-trip trading; for ``h > 1`` only one in every
``h`` consecutive predictions is consumed (the others are skipped).
This is the simplest honest design — it avoids the book-keeping of
overlapping positions while preserving leakage safety.

Costs
-----
Round-trip transaction cost ``c_bps`` (basis points) is deducted from
each closed trade. Slippage and market impact are bundled into
``c_bps``. The default is 10 bps; sensitivity sweeps at 0 / 5 / 25
bps are reported. For an h=1 active strategy on a model with
AUC ≈ 0.50, even 10 bps × 252 trades / year is enough to make the
net P&L solidly negative — that is the point of including this
analysis.

This module does not model: market hours, partial fills, borrow
costs on the short leg, or financing on the long leg.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Per-(model, horizon, cost) backtest output.

    All return-like fields are decimals (0.10 = 10%).
    """
    ticker: str
    model: str
    horizon: int
    cost_bps: float
    strategy: str

    n_trades: int
    n_long: int
    n_short: int
    hit_rate: float

    # Cumulative performance
    gross_total_return: float
    net_total_return: float

    # Annualised
    gross_ann_return: float
    net_ann_return: float
    ann_volatility: float
    gross_sharpe: float
    net_sharpe: float

    # Risk
    max_drawdown: float

    # Trade-level
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    avg_trade_return: float

    # Turnover (round-trips per year)
    turnover_per_year: float

    # Benchmarks computed on the same date span
    buy_and_hold_total_return: float
    buy_and_hold_ann_return: float
    buy_and_hold_sharpe: float

    # Time-series (kept as DataFrame, not flattened to scalars)
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def realize_strategy(
    predictions: pd.DataFrame,
    prices: pd.Series,
    *,
    horizon: int,
    cost_bps: float = 10.0,
    strategy: str = "sign",
    threshold: float = 0.0,
    ticker: str = "",
    model: str = "",
) -> BacktestResult:
    """Convert OOF predictions into a realised P&L and compute metrics.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must contain columns ``date``, ``y_pred``, optionally
        ``y_proba``. One row per prediction date.
    prices : pd.Series
        Daily close prices indexed by trading date. Must cover every
        ``predictions["date"] + h trading days``.
    horizon : int
        Forecasting horizon ``h``.
    cost_bps : float
        Round-trip transaction cost in basis points (default 10 bps).
    strategy : {"sign", "threshold"}
        ``"sign"`` uses ``y_pred`` directly. ``"threshold"`` requires a
        ``y_proba`` column and only trades when ``|y_proba - 0.5| >
        threshold``.
    threshold : float
        Conviction threshold (decimal). Ignored unless ``strategy =
        "threshold"``.

    Returns
    -------
    BacktestResult
    """
    if strategy not in {"sign", "threshold"}:
        raise ValueError(f"strategy must be 'sign' or 'threshold', got {strategy!r}")

    df = predictions[["date", "y_pred"]].copy()
    # Predictions CSVs can carry mixed-timezone date strings (some entries
    # naive, others ISO-with-offset). Coerce to UTC, strip tz, then
    # normalise to midnight so lookups against a tz-naive daily price
    # index succeed.
    df["date"] = (
        pd.to_datetime(df["date"], utc=True, errors="coerce")
          .dt.tz_localize(None).dt.normalize()
    )
    if df["date"].isna().any():
        raise ValueError("predictions['date'] contains unparseable values")
    df = df.sort_values("date").reset_index(drop=True)

    # Normalise prices index to midnight tz-naive so the get_indexer below
    # matches even when yfinance returns a tz-aware index.
    if isinstance(prices.index, pd.DatetimeIndex):
        idx_norm = prices.index
        if idx_norm.tz is not None:
            idx_norm = idx_norm.tz_convert(None) if idx_norm.tz is not None else idx_norm
            idx_norm = idx_norm.tz_localize(None)
        idx_norm = idx_norm.normalize()
        prices = pd.Series(prices.values, index=idx_norm, name=prices.name)

    if strategy == "threshold":
        if "y_proba" not in predictions.columns:
            raise ValueError("threshold strategy requires y_proba column")
        df["y_proba"] = predictions["y_proba"].values
        conviction = (df["y_proba"] - 0.5).abs() > threshold
        df["position"] = np.where(conviction, df["y_pred"], 0)
    else:
        df["position"] = df["y_pred"]

    # Non-overlapping h-day holds: take every h-th row, starting at row 0.
    # For h=1 this is every row (daily trading).
    df = df.iloc[::horizon].reset_index(drop=True)

    # Look up entry/exit prices.
    full_index = pd.Index(prices.index)
    entry_pos = full_index.get_indexer(df["date"])
    if (entry_pos < 0).any():
        missing = df.loc[entry_pos < 0, "date"].tolist()
        raise ValueError(f"prediction dates not in prices index: {missing[:5]}")
    exit_pos = entry_pos + horizon

    # Drop trades whose exit would overrun the price series.
    valid = exit_pos < len(full_index)
    df = df.loc[valid].reset_index(drop=True)
    entry_pos = entry_pos[valid]
    exit_pos = exit_pos[valid]

    df["entry_date"] = prices.index[entry_pos]
    df["exit_date"] = prices.index[exit_pos]
    df["entry_price"] = prices.values[entry_pos]
    df["exit_price"] = prices.values[exit_pos]

    # Gross trade return = position * (P_exit / P_entry - 1)
    df["gross_return"] = df["position"] * (df["exit_price"] / df["entry_price"] - 1.0)

    # Cost: per-trade cost is c_bps/1e4 round-trip; zero-position trades pay nothing.
    cost = cost_bps / 1e4
    df["cost"] = np.where(df["position"] != 0, cost, 0.0)
    df["net_return"] = df["gross_return"] - df["cost"]

    # Compounded equity curves on the closed-trade sequence.
    df["gross_equity"] = (1.0 + df["gross_return"]).cumprod()
    df["net_equity"]   = (1.0 + df["net_return"]).cumprod()

    # Metrics
    n_trades = int((df["position"] != 0).sum())
    n_long = int((df["position"] == 1).sum())
    n_short = int((df["position"] == -1).sum())

    closed = df[df["position"] != 0]
    if n_trades == 0:
        # Degenerate: no trades. Return zeros with correct schema.
        return _empty_result(
            ticker=ticker, model=model, horizon=horizon,
            cost_bps=cost_bps, strategy=strategy,
            prices=prices, df=df,
        )

    wins = closed["net_return"] > 0
    hit_rate = float(wins.mean())
    avg_win = float(closed.loc[wins, "net_return"].mean()) if wins.any() else 0.0
    avg_loss = float(closed.loc[~wins, "net_return"].mean()) if (~wins).any() else 0.0
    avg_trade = float(closed["net_return"].mean())

    gross_total = float(df["gross_equity"].iloc[-1] - 1.0)
    net_total   = float(df["net_equity"].iloc[-1] - 1.0)

    # Annualisation: trades-per-year proxy = TRADING_DAYS_PER_YEAR / horizon.
    trades_per_year = TRADING_DAYS_PER_YEAR / horizon
    if n_trades > 0:
        gross_ann = (1.0 + gross_total) ** (trades_per_year / n_trades) - 1.0
        net_ann   = (1.0 + net_total)   ** (trades_per_year / n_trades) - 1.0 \
                    if (1.0 + net_total) > 0 else float("nan")
    else:
        gross_ann = net_ann = 0.0

    ann_vol = float(closed["net_return"].std(ddof=1) * np.sqrt(trades_per_year)) \
              if n_trades >= 2 else 0.0
    gross_sharpe = float((closed["gross_return"].mean() * trades_per_year)
                         / (closed["gross_return"].std(ddof=1) * np.sqrt(trades_per_year))) \
                   if n_trades >= 2 and closed["gross_return"].std(ddof=1) > 0 else 0.0
    net_sharpe   = float((avg_trade * trades_per_year)
                         / ann_vol) if ann_vol > 0 else 0.0

    # Max drawdown on the net equity curve.
    eq = df["net_equity"].values
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    max_dd = float(dd.min())

    # Buy-and-hold benchmark over the SAME date span as the strategy.
    bh_start = df["entry_date"].iloc[0]
    bh_end   = df["exit_date"].iloc[-1]
    bh_ret_series = prices.loc[bh_start:bh_end].pct_change().dropna()
    if len(bh_ret_series) > 0:
        bh_total = float(prices.loc[bh_end] / prices.loc[bh_start] - 1.0)
        bh_n_days = (bh_end - bh_start).days
        bh_ann = (1.0 + bh_total) ** (365.25 / max(bh_n_days, 1)) - 1.0 if bh_total > -1 else float("nan")
        bh_vol = float(bh_ret_series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        bh_sharpe = float((bh_ret_series.mean() * TRADING_DAYS_PER_YEAR) / bh_vol) if bh_vol > 0 else 0.0
    else:
        bh_total = bh_ann = bh_sharpe = 0.0

    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    equity_curve = df[["entry_date", "exit_date", "position",
                       "gross_return", "net_return",
                       "gross_equity", "net_equity"]].copy()
    trades = df.copy()

    return BacktestResult(
        ticker=ticker, model=model, horizon=horizon,
        cost_bps=cost_bps, strategy=strategy,
        n_trades=n_trades, n_long=n_long, n_short=n_short,
        hit_rate=hit_rate,
        gross_total_return=gross_total, net_total_return=net_total,
        gross_ann_return=gross_ann, net_ann_return=net_ann,
        ann_volatility=ann_vol,
        gross_sharpe=gross_sharpe, net_sharpe=net_sharpe,
        max_drawdown=max_dd,
        avg_win=avg_win, avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        avg_trade_return=avg_trade,
        turnover_per_year=float(trades_per_year),
        buy_and_hold_total_return=bh_total,
        buy_and_hold_ann_return=bh_ann,
        buy_and_hold_sharpe=bh_sharpe,
        equity_curve=equity_curve,
        trades=trades,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result(*, ticker, model, horizon, cost_bps, strategy,
                  prices, df) -> BacktestResult:
    """Return zero-filled result for a degenerate (no-trade) run."""
    return BacktestResult(
        ticker=ticker, model=model, horizon=horizon,
        cost_bps=cost_bps, strategy=strategy,
        n_trades=0, n_long=0, n_short=0, hit_rate=0.0,
        gross_total_return=0.0, net_total_return=0.0,
        gross_ann_return=0.0, net_ann_return=0.0,
        ann_volatility=0.0, gross_sharpe=0.0, net_sharpe=0.0,
        max_drawdown=0.0,
        avg_win=0.0, avg_loss=0.0, win_loss_ratio=float("nan"),
        avg_trade_return=0.0,
        turnover_per_year=TRADING_DAYS_PER_YEAR / horizon,
        buy_and_hold_total_return=0.0,
        buy_and_hold_ann_return=0.0,
        buy_and_hold_sharpe=0.0,
        equity_curve=df[["entry_date" if "entry_date" in df.columns else "date"]].head(0),
        trades=df.head(0),
    )


def metrics_to_row(result: BacktestResult) -> dict:
    """Flatten BacktestResult to a single CSV row (no pandas in values)."""
    d = asdict(result)
    # Drop the time-series fields — they go to a separate CSV.
    d.pop("equity_curve", None)
    d.pop("trades", None)
    return d


# ---------------------------------------------------------------------------
# Sweep convenience
# ---------------------------------------------------------------------------

def sweep_backtests(
    predictions: pd.DataFrame,
    prices: pd.Series,
    *,
    horizon: int,
    cost_bps_grid: Sequence[float] = (0.0, 5.0, 10.0, 25.0),
    strategies: Sequence[str] = ("sign",),
    threshold: float = 0.0,
    ticker: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run :func:`realize_strategy` for every (model, cost, strategy) tuple.

    Returns
    -------
    metrics_df : pd.DataFrame
        One row per (model, cost_bps, strategy) with all scalar metrics.
    equity_df : pd.DataFrame
        Long-form equity curves with the same key columns; suitable for
        plotting or join-with-prices.
    """
    rows: list[dict] = []
    eqs: list[pd.DataFrame] = []
    for model_name, sub in predictions.groupby("model"):
        for c in cost_bps_grid:
            for strat in strategies:
                res = realize_strategy(
                    sub, prices,
                    horizon=horizon, cost_bps=c,
                    strategy=strat, threshold=threshold,
                    ticker=ticker, model=model_name,
                )
                rows.append(metrics_to_row(res))
                eq = res.equity_curve.copy()
                eq["ticker"] = ticker
                eq["model"] = model_name
                eq["horizon"] = horizon
                eq["cost_bps"] = c
                eq["strategy"] = strat
                eqs.append(eq)
    metrics_df = pd.DataFrame(rows)
    equity_df = pd.concat(eqs, ignore_index=True) if eqs else pd.DataFrame()
    return metrics_df, equity_df
