# Economic Backtest Report — AAPL, h=1/5/15

**Branch:** `qf-05-economic-backtest`
**Audit cross-reference:** critical issue C-19 — *No realised P&L,
Sharpe, or transaction-cost backtest.*
**Status:** module + tests shipped; backtest run on every horizon
for which leakage-safe predictions exist; results saved.

---

## 1. Why this exists

The audit's verification report listed C-19 as a critical gap: the
paper measured directional accuracy but never translated predictions
into a realised P&L. For Quantitative Finance an accuracy claim is
incomplete without the round-trip cost picture, because

> $\text{net annual P\&L} \;=\;
> \text{gross edge} - n_\text{trades} \cdot c_\text{round-trip}$

and at the AUC ≈ 0.50 levels documented in the leakage-safe
nested-CV addendum (`07_nested_cv_first_run.md`), the gross edge is
small. Whether anything survives the cost wedge is the open
question.

This addendum closes C-19 for the AAPL single-asset case. The
panel-wide backtest is gated on the panel-wide nested-CV sweep and
remains future work.

---

## 2. Strategy specification

For every leakage-safe OOF prediction
$(t, \text{model}, \hat y \in \{-1, +1\}, p)$ at horizon $h$:

- **Sign strategy** (default): position $\pi_t = \hat y$. Long if
  the model predicts up, short if it predicts down. The position is
  held from the close of day $t$ to the close of day $t+h$.
- **Threshold strategy**: position only when $|p - 0.5| > \delta$;
  on low-conviction predictions the strategy stays in cash.

Trades are **non-overlapping at horizon $h$**. The driver consumes
predictions at indices $0, h, 2h, \dots$ from the chronologically
sorted OOF series; the others are ignored. For $h=1$ this is daily
round-trip trading; for $h=15$ it is one round-trip per 15 trading
days. This is the simplest honest design — it avoids modelling
overlapping books and keeps the leakage safety inherited from the
nested-CV pipeline intact.

**Frictions.** Per-trade round-trip cost $c_\text{bps}$ deducted on
every closed position. Default $10$~bps; sensitivity grid
$\{0, 5, 10, 25\}$~bps. Slippage and market impact are bundled into
$c_\text{bps}$.

**Benchmarks.** Buy-and-hold AAPL over the same date span as the
strategy's first-entry to last-exit window.

**Not modelled.** Borrow costs on the short leg; financing on the
long leg; partial fills; liquidity caps; bid-ask spread beyond the
flat $c_\text{bps}$.

---

## 3. Results

Source: `results/backtest_metrics.csv`, run timestamp in
`results/backtest_run_snapshot.json`.

### 3.1 Buy-and-hold benchmark

The same AAPL price series at the same date span gives the
benchmark every active strategy must clear:

| Horizon span (start → end) | B&H total return | B&H ann.\ return |
|---|---|---|
| 2020-02-11 → 2024-12-30 ($h=1$) | 226 % | **26.5 %** |
| 2020-02-13 → 2024-12-23 ($h=5$) | 226 % | **27.0 %** |
| 2020-03-04 → 2024-12-26 ($h=15$) | 217 % | **26.3 %** |

### 3.2 $h=1$ active trading (1 229 round-trips)

| Model | Hit rate | Gross ann.\ | Net @ 0 bps | Net @ 5 bps | Net @ 10 bps | Net @ 25 bps |
|---|---|---|---|---|---|---|
| **Random Forest** | 51.2 % | **+18.3 %** | +18.3 % | +4.3 % | **−8.1 %** | −37.1 % |
| XGBoost | 48.7 % | −0.5 % | −0.5 % | −12.3 % | −22.7 % | −47.1 % |
| LightGBM | 48.2 % | −3.7 % | −3.7 % | −15.1 % | −25.2 % | −48.8 % |

**Reading.**

- Even at zero cost, only Random Forest produces a positive gross
  return, and at +18.3 %/yr it underperforms buy-and-hold by 8 pp.
- At a realistic 10 bps round-trip, Random Forest's net return is
  *negative*: the 1 229 daily round-trips × 10 bps ≈ 12.3 pp of
  annual frictional drag overwhelms the small gross edge.
- At 25 bps every model loses roughly 40 % per year.
- All three models have hit rates within the [48 %, 52 %] band that
  the leakage-safe AUC ≈ 0.50 already predicted. There is no
  directional skill.

### 3.3 $h=5$ active trading (245 round-trips)

| Model | Hit rate | Gross ann.\ | Net @ 10 bps | Max DD |
|---|---|---|---|---|
| Random Forest | 52.2 % | −5.0 % | −9.7 % | −67 % |
| XGBoost | 47.4 % | −17.1 % | −21.2 % | −82 % |
| LightGBM | 49.8 % | −10.2 % | −14.6 % | −69 % |

Every model loses money even gross. The 5-day horizon does not
unlock any signal that survives realistic execution.

### 3.4 $h=15$ active trading (81 round-trips)

| Model | Hit rate | Gross ann.\ | Net @ 10 bps | Net @ 25 bps | Net Sharpe @ 10 bps |
|---|---|---|---|---|---|
| **XGBoost** | 54.3 % | **+11.5 %** | **+9.7 %** | +7.0 % | 0.46 |
| LightGBM | 53.1 % | +9.1 % | +7.3 % | +4.6 % | 0.39 |
| Random Forest | 50.6 % | +8.3 % | +6.5 % | +3.9 % | 0.36 |

**Reading.**

- $h=15$ is the only configuration where active trading produces
  positive net returns at retail-friendly cost levels.
- The hit rate is mildly above 50 % across all three models, and
  the round-trip count is small enough (81) that even 25 bps adds
  only ~2 pp of annual drag.
- Yet the best model nets **+9.7 %/yr versus 26.3 %/yr buy-and-hold**.
  The strategy captures ~37 % of the passive benchmark's return
  while taking on a −36 % max drawdown — clearly worse than
  unconditional long exposure.
- The §10.4 caveat in the paper addendum applies here: at $h=15$
  the elevated accuracy partly reflects class-imbalance exploitation
  (15-day cumulative returns are right-skewed in 2020--2024). The
  +9.7 %/yr is therefore consistent with a model predicting "up" most
  of the time and benefiting from the bull-market drift, not with a
  stable directional edge.

### 3.5 Tabular summary

The single-line take-away across the 36-cell grid:

- $0$ active strategies $\succ$ buy-and-hold at any horizon and any cost.
- $1$ strategy (RF, $h=1$, $0$ bps) edges close to B&H gross but
  loses to it.
- All three strategies at $h=15$ produce positive *absolute* net
  returns at $\le 10$ bps — but each is dominated by the buy-and-hold
  benchmark by 16–18 pp/yr.

---

## 4. Implications for the paper

The §10 addendum already corrected the *accuracy* claims. The
addendum could be strengthened by adding a one-paragraph footnote
that addresses the *economic* claim. Suggested text (one paragraph):

> Even where the leakage-safe nested CV produces above-50 % accuracy
> ($h=15$, hit rate 53–54 %), translating those predictions into a
> realised P&L net of 10 bps round-trip costs yields at best
> +9.7 %/yr (XGBoost), against a buy-and-hold benchmark of 26.3 %/yr
> over the same span. At $h=1$ the picture is starker: every active
> strategy nets *negative* annualised return at any cost level
> $\ge 5$ bps, because daily round-trips at the model's hit rate of
> roughly 51 % cannot overcome the cost wedge. The directional model
> does not survive contact with execution friction; the
> economic-significance claim that closes the original Section~4.6
> ("portfolio-level aggregation improves accuracy by 7–13 pp")
> therefore needs explicit qualification: *accuracy* improves;
> *realised P&L net of costs* does not necessarily follow.

The author can drop this paragraph into the addendum's §10.6
"What survives" subsection, or into a fresh §10.8.

---

## 5. Module + tests

`src/backtest.py` — `realize_strategy`, `BacktestResult`,
`sweep_backtests`, plus `metrics_to_row` for CSV serialisation.
Strategy-spec details in §2 above; the implementation forces every
prediction date to be present in the prices index and refuses to
silently drop unobserved observations.

`tests/test_backtest.py` (11 cases, all passing) covers:

| # | Property |
|---|---|
| 1 | All-long on a strictly rising series ≈ buy-and-hold (gross). |
| 2 | All-short on a rising series → loss. |
| 3 | Perfect predictor dominates buy-and-hold (gross). |
| 4 | Cost is exactly subtracted: per-trade NET @ 0 bps − per-trade NET @ 10 bps = 10/10⁴. |
| 5 | Threshold strategy filters low-conviction signals. |
| 6 | Equity curve is finite and positive (no NaN/Inf). |
| 7 | Non-overlapping execution: $n_\text{trades} = $ count of $kh$ such that $kh + h <$ index length, parametrised over $h \in \{1,2,5,10\}$. |
| 8 | `sweep_backtests` end-to-end smoke. |

```bash
python -m pytest tests/test_backtest.py -v   # 11 passed in 0.66s
python -m pytest                              # 59 passed (full suite)
```

---

## 6. Reproducing this run

The exact command that produced the committed CSVs:

```bash
git checkout qf-05-economic-backtest
python scripts/run_backtest.py \
    --horizons 1 5 15 \
    --costs-bps 0 5 10 25 \
    --strategy sign
```

Outputs (committed under `results/`):

- `backtest_metrics.csv` — 36 rows, schema in §3.
- `backtest_equity.csv` — long-form equity curves; one row per
  closed trade per (model, horizon, cost) cell.
- `backtest_run_snapshot.json` — timestamps, cost grid, horizon
  list, and the start/end dates used to fetch AAPL OHLCV.

Sources for the OOF predictions:

- `results/nested_cv_predictions.csv`        ($h=1$, from `qf-04`)
- `results/nested_cv_h5/nested_cv_predictions.csv`  ($h=5$, from `qf-08`)
- `results/nested_cv_h15/nested_cv_predictions.csv` ($h=15$, from `qf-08`)

The script re-fetches AAPL OHLCV from yfinance for the price series;
this is the same source as the nested-CV pipeline, so the prices are
internally consistent.

---

## 7. What this run is NOT

- Not panel-wide. Only AAPL is covered; the other 24 stocks have no
  OOF predictions yet.
- Not multi-strategy. The threshold strategy is implemented and
  tested but the headline run uses only the sign strategy. A
  threshold sweep at $\delta \in \{0.05, 0.10, 0.15\}$ is a one-line
  CLI change but is deferred to keep the headline summary readable.
- Not regime-decomposed. The 2020-2024 sample contains the COVID
  crash, the 2022 selloff, and the 2023-2024 AI rally; performance
  may differ across these. A sub-period decomposition is a natural
  follow-up.
- Not benchmarked against alternatives. The persistence baseline
  ("predict tomorrow = today") quoted in the paper at 63.2 % was
  never re-evaluated under nested CV; doing so here would require
  generating persistence predictions and is left for the
  statistical-inference branch (audit C-22).
- Not stress-tested for short-borrow availability. Some $h=1$ days
  have negative positions (short AAPL) that may not be executable in
  practice without a stable borrow rate.

---

## 8. Final verdict

**The leakage-safe predictions on AAPL at every tested horizon
underperform a passive buy-and-hold position once realistic
transaction costs are deducted.** At $h=1$ the strategy's net P&L is
negative for every model at every cost ≥ 5 bps; at $h=15$ the best
strategy nets +9.7 %/yr against a 26.3 %/yr buy-and-hold benchmark.

This is the empirical answer to the C-19 audit critical: under the
leakage-safe protocol, the accuracy figures in the paper do not
correspond to economically meaningful trading returns.
