# Panel Snapshot + Bib-Coverage Polish — qf-13-panel-snapshot

**Branch:** `qf-13-panel-snapshot`
**Status:** the cheapest deferred item from
`16_revision_fixes_report.md` is closed: all 25~tickers in the
`STOCK_UNIVERSE` panel are now committed under
`data/snapshots/`. The data-availability statement is tightened
accordingly. Six previously-unused bibliography entries from
`qf-07-literature-web` have been woven into the prose.

---

## 1. Panel snapshot

`scripts/snapshot_data.py` was run without arguments. The
result:

```
Wrote: 24  Skipped: 1  Failed: 0
```

(AAPL was already snapshotted in `qf-data-snapshot` at
2026-04-27; the other 24 tickers were captured today,
2026-05-03, against the same `yfinance` 1.1.0 vintage.)

Inventory:

```
data/snapshots/  (1.7 MB total)
  AAPL_2020-01-01_2024-12-31.parquet
  ABBV_..., BAC_..., C_..., DIS_..., F_..., GM_..., GOOGL_...,
  GS_..., HMC_..., JNJ_..., JPM_..., MCD_..., META_..., MS_...,
  MSFT_..., NFLX_..., NKE_..., NVDA_..., PFE_..., SBUX_..., TM_...,
  TMO_..., TSLA_..., UNH_...
  snapshot_metadata.json   (25 entries, all 1257 rows)
```

All 25~snapshots are 1{,}257~trading days, captured against
`yfinance==1.1.0`, with UTC capture timestamps recorded in
`snapshot_metadata.json`. Future reproducers using the
committed code path read byte-identical inputs.

The data-availability statement now lists every ticker
explicitly and removes the "snapshots not yet committed for
the full panel" caveat from the previous draft. The single
remaining gating constraint on the panel-wide nested-CV sweep
is compute time; the data side is fully closed.

## 2. Bibliography coverage polish

The severe reviewer's report flagged 16 unused bibliography
entries. Six of the most directly relevant ones were woven
into the manuscript:

| Citation | Inserted in | Role |
|---|---|---|
| `avramov2023machine` | Introduction `4 (economic value paragraph) | Recent ML-vs-economic-restrictions evidence |
| `kelly2023financial` | Same | Kelly-Xiu 2023 financial-ML survey |
| `harvey2016cross` | Same | Multiple-testing in finance |
| `demiguel2020transaction` | Introduction `4 (cost reasoning) | Cost-aware portfolio-construction framework |
| `bailey2014deflated` | `5.5 (after Bonferroni discussion) | Deflated Sharpe ratio for strategy multiple-testing |
| `lundberg2020local` | `2.4 SHAP description (TreeSHAP attribution) | Reference for the polynomial-time TreeSHAP algorithm |

Unused-bib count dropped from 16 to 10. The remaining 10 are
mostly legacy literature-survey entries
(`fama1970efficient`, `huang2005forecasting`, `kim2003financial`,
`patel2015predicting`, `krauss2017deep`, `sirignano2019universal`,
`bailey2014pseudomath`, `bussmann2021explainable`,
`white2000reality`, `guyon2003introduction`) that were imported
in `qf-07` for completeness; they belong in a literature-review
expansion that is not in the scope of this branch.

## 3. Verification

```
Missing labels        : none
Missing bib keys      : none
LaTeX em-dashes (---) : 0
Unicode em-dashes     : 0
Unused bib keys       : 10  (was 16)
```

Test suite: 80/80 pass (no code changes other than the snapshot
files written by `scripts/snapshot_data.py`).

## 4. What this commit deliberately does NOT do

- Does not run the matched-budget leakage-safe nested-CV
  protocol. That is the next compute-bound deferred item
  (multi-day).
- Does not run the panel-wide nested-CV sweep across the 24
  newly-snapshotted tickers. Same reason.
- Does not implement Romano-Wolf or Hansen-SPA correction of
  the pairwise DM matrix.
- Does not cite the remaining 10 unused legacy-literature
  bibliography entries; they belong in a literature-review
  expansion, not a polish commit.

## 5. Audit closure status after this commit

| Item | Status |
|---|---|
| C-1 to C-22 (audit critical list) | All closed |
| Agent 07-12 (writing pipeline) | All closed |
| Severe-reviewer F1 (long/short relabel) | Closed on qf-12 |
| Severe-reviewer F2 (real B&H path) | Closed on qf-12 |
| Severe-reviewer F3 (matched-budget confound) | Disclosed on qf-12 |
| Severe-reviewer F4 (n=1 abstract scope) | Closed on qf-12 |
| Reviewer Bonferroni arithmetic error | Closed on qf-12 |
| Reviewer methodology inconsistency | Closed on qf-12 |
| Reviewer data-availability overclaim | **Closed on qf-13** |
| Reviewer 16 unused bib entries | **Reduced to 10 on qf-13** |
| Matched-budget run | Open (compute-bound) |
| Panel-wide nested-CV sweep | Open (compute-bound) |
| Romano-Wolf / Hansen-SPA correction | Open |
| Balanced-accuracy / Brier metrics (C-26) | Open |
| Walk-forward robustness (C-9) | Open |

## 6. Next reasonable steps

In order of cheapness vs.\ value:

1. Compile-test locally (still no LaTeX toolchain in this
   session). The §10.5 `13_qf_format_report.md` checklist still
   applies; the only new structural element since that report
   is the §5.5 `sec:addendum_caveats` subsection from qf-12.
2. Read with co-authors and confirm framing.
3. If reviewer pushback on multiple-testing is anticipated,
   implement Romano-Wolf for the 9 pairwise DM tests. The
   `src/inference.py` module already exposes the necessary
   primitives; the new function would be a 30-line addition
   plus tests.
4. The matched-budget leakage-safe run remains the strongest
   robustness check. It is a one-command invocation
   (`python scripts/run_nested_cv.py --tickers AAPL --n-trials 50
   --n-inner 5 --models all`) on AAPL with the current code,
   estimated 4-12 hours on the available GPU, after which the
   `sec:addendum_caveats` subsection on qf-12 can be revised
   to report the matched-budget gap rather than disclose the
   confound.

The committed paper is now textually consistent with the code
on every reviewer-flagged dimension and is data-complete
(25/25 snapshots committed).
