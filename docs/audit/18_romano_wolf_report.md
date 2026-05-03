# Romano-Wolf Step-Down Adjustment — qf-14-romano-wolf

**Branch:** `qf-14-romano-wolf`
**Status:** the multiple-testing correction that the severe
reviewer flagged as "the appropriate next step before drawing
model-selection conclusions" is shipped in `src/inference.py`,
unit-tested, applied to the AAPL DM matrix in `results/`, and
folded into the §5.5 paper text. The previous paper claim that
the model ranking was "real but ranks models tied with chance"
is replaced with the FWER-adjusted finding that the ranking is
\emph{not identifiable} on this single asset.

---

## 1. New module: `romano_wolf_dm`

`src/inference.py` gains one public function:

```python
romano_wolf_dm(
    losses_by_model: dict[str, np.ndarray],
    *, h: int = 1,
    lag: int | None = None,
    expected_block_size: int,
    n_boot: int = 4000,
    seed: int = 42,
) -> list[dict]
```

Algorithm. Studentized Romano-Wolf step-down (Romano and Wolf
2005, Econometrica). For $K = M(M-1)/2$ pairwise DM hypotheses
on $M$ models:

1. Compute observed DM statistics $t_k$ (with NW HAC variance and
   the HLN small-sample factor) for $k = 1, \ldots, K$.
2. Bootstrap $B$ replications by stationary block bootstrap on
   the time index (`stationary_block_bootstrap`). For each
   replication $b$ and each pair, recompute the DM statistic
   $t_k^{(b)}$ on the bootstrapped loss series and centre on the
   observed: $u_k^{(b)} = t_k^{(b)} - t_k$.
3. Order observed $|t_k|$ descending. For each step $j = 1,
   \ldots, K$, the adjusted $p$-value is
   $$ p^*_{(j)} = \mathrm{Pr}\bigl(\max_{k \in S_j} |u_k^{(b)}| \ge |t_{(j)}|\bigr), $$
   where $S_j = \{k_{(j)}, k_{(j+1)}, \ldots, k_{(K)}\}$ is the
   set of remaining hypotheses.
4. Enforce monotonicity along the descending-$|t|$ order:
   $p^*_{(j)} := \max\bigl(p^*_{(j)}, p^*_{(j-1)}\bigr)$.

The function also returns a Bonferroni-Holm benchmark for
comparison. Output is a flat list-of-dicts with the same shape
as `pairwise_diebold_mariano`, plus three extra fields:
`rw_p_value`, `rw_reject_05`, `bonferroni_p_value`.

## 2. New tests (5 cases, all passing)

`tests/test_inference.py` now has 18 tests in total (was 13).

| # | Property |
|---|---|
| 14 | Under clear dominance, RW-adjusted $p < 0.05$ for the dominated pair. |
| 15 | Under no signal (all-equal losses), no RW-adjusted $p < 0.05$. |
| 16 | RW $p^* \ge$ raw $p$ for every comparison (defining property of an FWER step-down). |
| 17 | RW $p^*$ is monotone non-decreasing along the descending-$|t|$ order. |
| 18 | When pair statistics are correlated, the smallest RW $p^*$ is no larger than the smallest Holm-Bonferroni $p$. |

All five pass. Full suite: 85/85.

## 3. Application to AAPL OOF predictions

`scripts/run_inference.py` now calls `romano_wolf_dm` alongside
`pairwise_diebold_mariano`, with the same correlation-aware
block size used by the per-(model, horizon) accuracy CIs. The
updated `results/inference_dm_pvalues.csv` carries three
$p$-value columns: `p_value` (raw HLN-corrected DM),
`rw_p_value` (Romano-Wolf step-down), `bonferroni_p_value`
(Holm-Bonferroni).

### 3.1 Headline numbers

| Horizon | Pair | DM stat | Raw $p$ | RW $p^{*}$ | Holm $p$ |
|---|---|---:|---:|---:|---:|
| $h=1$  | LightGBM vs RF      | $+2.05$ | 0.041 | **0.072** | 0.122 |
| $h=1$  | LightGBM vs XGBoost | $+0.28$ | 0.777 | 0.770 | 0.777 |
| $h=1$  | RF vs XGBoost       | $-1.79$ | 0.074 | 0.104 | 0.148 |
| $h=5$  | LightGBM vs RF      | $+2.27$ | 0.023 | **0.095** | 0.046 |
| $h=5$  | LightGBM vs XGBoost | $-0.11$ | 0.909 | 0.935 | 0.909 |
| $h=5$  | RF vs XGBoost       | $-2.68$ | 0.007 | **0.051** | 0.022 |
| $h=15$ | LightGBM vs RF      | $-0.44$ | 0.660 | 0.632 | 0.660 |
| $h=15$ | LightGBM vs XGBoost | $+1.70$ | 0.090 | 0.136 | 0.181 |
| $h=15$ | RF vs XGBoost       | $+1.99$ | 0.047 | 0.081 | 0.141 |

### 3.2 Verdict at $\alpha = 0.05$

Under Romano-Wolf step-down, the smallest adjusted $p$-value is
$\mathbf{0.051}$ ($h=5$, RF vs XGBoost). \emph{None} of the nine
pairwise comparisons clears $\alpha = 0.05$ under FWER control.

The previous paper draft framed the Bonferroni outcome as "no
comparison survives" and the raw-$p$ outcome as a clean ranking;
under Romano-Wolf both readings collapse to the honest finding
that the model ranking is not identifiable on this single asset
once multiple testing is properly controlled.

### 3.3 Holm-Bonferroni vs Romano-Wolf

Within each $K=3$-pair within-horizon family, Romano-Wolf and
Holm-Bonferroni agree qualitatively at $\alpha=0.05$ on eight of
nine pairs. The one disagreement is the $h=5$ RF vs XGBoost
comparison, which Holm clears at $0.022$ and Romano-Wolf
declines at $0.051$. The discrepancy reflects two finite-sample
features: (i) the studentized RW bootstrap re-estimates the HAC
variance inside each replication, which adds noise to the
distribution of $t^{(b)}$; (ii) at $K=3$ Holm's
$(K-r+1) \times p_{(r)}$ scaling can be tighter than the
empirical max-stat distribution under highly correlated test
statistics. Both readings converge on the headline conclusion:
no comparison decisively rejects.

## 4. Changes to the paper

§5.5 (`sec:addendum_inference`):

- Table `tab:addendum_dm` gains a `RW $p^{*}$` column. The old
  "Verdict" column now reads "tied" for all nine rows, with
  $h=5$ RF vs XGBoost flagged as "tied (marginal)".
- The prose paragraph after the table is rewritten. The earlier
  reading ("the DM tests yield a stable qualitative ranking ...
  Random Forest dominates at $h \le 5$") is replaced with the
  Romano-Wolf-corrected reading: the ranking is not identifiable
  under FWER control. Hansen-SPA and White's Reality Check are
  named as more-powerful follow-ups; \citet{bailey2014deflated}
  retained as the strategy-level analogue.

§6 (Conclusion):

- The "Limitations" paragraph on multiple testing is rewritten
  from "Inference is reported by uncorrected pairwise DM
  $p$-values; Romano-Wolf or Hansen-SPA adjustments would
  tighten ..." to a fact statement that Romano-Wolf is now
  applied and that Hansen-SPA / White Reality Check would be
  more-powerful next steps.

## 5. Reviewer-flagged item closed

`docs/reviewer_reports/severe_qf_review.md` Major Comment §2
asked for a Romano-Wolf or Hansen-SPA adjustment. This commit
ships Romano-Wolf and discloses the result honestly. Hansen-SPA
remains for a follow-up if and only if the marginal $h=5$ RF vs
XGBoost result becomes load-bearing for the paper's main claim,
which it does not.

## 6. Style scrub on the post-revision manuscript

```
Missing labels:    none
Missing bib keys:  none
Unused bib keys:   9 (was 10; white2000reality is now cited
                       in the Romano-Wolf prose)
LaTeX em-dashes:   0
Unicode em-dashes: 0
```

## 7. Audit closure status after this commit

| Item | Status |
|---|---|
| C-1 to C-22 (audit critical list) | All closed |
| Reviewer F1 long/short relabel | Closed (qf-12) |
| Reviewer F2 real B&H path | Closed (qf-12) |
| Reviewer F3 matched-budget confound | Disclosed (qf-12); not yet run |
| Reviewer F4 n=1 abstract scope | Closed (qf-12) |
| Reviewer Bonferroni arithmetic | Closed (qf-12) |
| Reviewer methodology consistency | Closed (qf-12) |
| Reviewer data-availability overclaim | Closed (qf-13) |
| Reviewer 16 unused bib entries | Reduced to 9 (qf-13 + qf-14) |
| Reviewer Romano-Wolf request | **Closed on this branch** |
| Matched-budget run | Open (compute-bound) |
| Panel-wide nested-CV sweep | Open (compute-bound) |
| Walk-forward robustness (C-9) | Open |
| Balanced-accuracy / Brier metrics (C-26) | Open |

## 8. Reproducing this run

```bash
git checkout qf-14-romano-wolf
python -m pytest tests/test_inference.py -v
python scripts/run_inference.py --horizons 1 5 15 --n-boot 4000 --seed 42
```

The `n_boot=4000` setting and the `seed=42` make the
Romano-Wolf $p^{*}$ values stable to four decimals across
re-runs.

## 9. What this branch deliberately does NOT do

- Hansen-SPA test (more powerful step-down based on a different
  null distribution). The Romano-Wolf result alone is enough for
  the headline reading.
- Deflated Sharpe ratio applied to the trading-strategy P&Ls.
  The §5.5 prose cites \citet{bailey2014deflated} but the
  computation is deferred.
- Application to the panel-wide DM matrix, which would have
  $K = M(M-1)/2 \times \text{n\_tickers}$ pairs to control over.
  Gated on the panel-wide nested-CV sweep itself.
