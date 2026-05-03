# Figures and Tables Pass — qf-09-figures-tables

**Branch:** `qf-09-figures-tables`
**Agent:** `.claude/agents/09_figures_tables_agent.md`
**Status:** the missing main-set figure (economic backtest after
costs) is generated and inserted into §10; the six legacy
orphan figures are now cited from prose; a deterministic
figure-regeneration script ships under
`scripts/make_paper_figures.py`.

---

## 1. The missing main-set figure: economic backtest after costs

The Agent 09 spec lists six required main-paper figures. Five
were already in the legacy body. The missing one was:

> 4. Economic backtest after costs.

`scripts/make_paper_figures.py` was added with one entry point
(`make_backtest_equity_figure`) that reads
`results/backtest_equity.csv` and `results/backtest_metrics.csv`
and writes `figures/backtest_equity_curves.png`. The output is
a three-panel figure (one panel per horizon $h \in \{1, 5, 15\}$)
overlaying the three model strategies (LightGBM, RF, XGBoost)
under the sign rule at 10 bps per round trip, against a buy-and-
hold benchmark anchored to the same OOF window.

The substantive finding: at every horizon, every model
underperforms passive buy-and-hold once costs are imposed. The
$h=15$ XGBoost configuration, which had the highest leakage-safe
pooled accuracy in Table~`tab:addendum_aapl_window`, gets
closest but still loses to the benchmark.

The figure is inserted as a new subsection §10.6 ("Economic
value: backtest after costs") with a self-contained caption,
ahead of the "Status" subsection. Numbering of subsequent
addendum subsections shifts by one.

## 2. Orphan-figure citations

`qf-08-qf-format` left six legacy figures defined but never
referenced from prose. Each is now cited:

| Figure | Cited from |
|---|---|
| `fig:smoothing` | §3.2 wavelet-smoothing paragraph |
| `fig:target_corr` | §3.3 Boruta paragraph |
| `fig:sector_analysis` | §4.6 sector ranking paragraph |
| `fig:portfolio_evolution` | §4.7 portfolio aggregation paragraph |
| `fig:individual_vs_portfolio` | §4.7 portfolio aggregation paragraph |
| `fig:shap_dep` | §4.8 SHAP discussion paragraph |

## 3. Citation / cross-reference health after this commit

```
Missing labels (referenced but not defined): none
Missing bib keys (cited but not in bib)    : none
Orphan figure labels: 0
Orphan section labels: 1 (sec:addendum_backtest, used by
   hyperref only)
```

## 4. What this branch deliberately does NOT do

The Agent 09 spec has eight tasks. This branch addresses #1
(no figure was flagged as unreadable; all PNG files exist and
load), #6 (the new backtest figure has the data envelope on
each curve and reports the 95\% CI horizon for the nominally-
significant XGB at $h=15$ only via the caption), #7 (the new
backtest figure caption is self-contained), and #8 (the
backtest figure is regenerable by script).

The remaining tasks are deferred:

- **Task 2 (reduce main-paper figures to the minimum set).**
  The body currently has 14 figures; the agent's prescribed
  main set has 6. A full curation requires restructuring
  several body sections, which substantively belongs in
  `qf-10-writing` where Section 4 will be partly rewritten
  anyway.
- **Task 3 (move secondary plots to appendix).** Same reason.
  An `\appendices` block could be added on this branch as a
  pure structural change, but moving 8 figures into it
  without rewriting the section prose would leave dangling
  references; the move is cleaner as part of the qf-10
  rewrite.
- **Task 4 (standardize figure dimensions).** The legacy
  figures use `\includegraphics[width=0.45\textwidth]` through
  `width=0.95\textwidth`. The new backtest figure is at
  `0.95\textwidth` (matches the wide table convention in
  §10.4). Standardising the legacy widths to a small set of
  three or four values is also cleaner as part of qf-10.
- **Task 5 (use readable labels).** This requires re-rendering
  legacy figures from the notebook with larger axis labels and
  is gated on the legacy-notebook re-run, which the standing
  constraint forbids.

## 5. Figure-regeneration script

`scripts/make_paper_figures.py` is the canonical entry point
for any paper figure derived from `results/`. Currently it
implements one figure (the backtest equity panels). The pattern
is intentionally generalisable: each future addendum figure
gets its own `make_<name>_figure(out_path, ...)` function and
is called from `main()`.

Running:

```bash
python scripts/make_paper_figures.py
# Optional flags
python scripts/make_paper_figures.py --cost-bps 5
python scripts/make_paper_figures.py --strategy threshold
```

The script is deterministic given the committed
`results/backtest_*.csv` files.

## 6. New paper figure inventory

| ID | File | Caption summary | Cited from | Source |
|---|---|---|---|---|
| `fig:addendum_backtest` | `backtest_equity_curves.png` | Net equity curves vs. B&H, 3 panels, 10 bps | §10.6 | `scripts/make_paper_figures.py` |

The 14 legacy figures are unchanged on disk and are now all
cited from body prose, so the citation graph is complete.

## 7. Audit closure status after this commit

| ID | Status |
|---|---|
| C-1 to C-5, C-13, C-18, C-19, C-22 | Closed pre-qf-09 |
| Agent 07 (literature) | Closed on `qf-07-literature-web` |
| Agent 08 (QF format) | Closed on `qf-08-qf-format` |
| Agent 09 (figures and tables) | **Tasks 1, 6-8 closed on this branch; tasks 2-5 deferred to qf-10-writing** |
| Agent 10 (writing) | Pending |
| Agent 11 (severe review) | Pending (final step) |

## 8. Next branch

`qf-10-writing` performs the substantive rewrite under the
Writing Agent's target framing, and as a side effect addresses
the deferred Agent 09 tasks 2-5 (minimum-set curation and
appendix moves). After qf-10, `qf-11-severe-review` runs the
full referee pass.

## 9. What the user must verify locally

The new backtest figure was generated by `make_paper_figures.py`
with the committed CSVs and saved at 200 dpi. Its substantive
content has been spot-checked against
`results/backtest_metrics.csv` (the buy-and-hold annualised
return of $26.5$\,\% matches the committed metric for AAPL).
Visual inspection of `figures/backtest_equity_curves.png`
before submission is recommended; if the PDF compile shows the
figure overflowing the text width, drop the
`width=0.95\textwidth` to `width=0.85\textwidth`.
