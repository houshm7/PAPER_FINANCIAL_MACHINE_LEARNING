# Revision Fixes — qf-12-revision-fixes

**Branch:** `qf-12-revision-fixes`
**Reviewer report:** `docs/reviewer_reports/severe_qf_review.md`
**Status:** the four fatal flaws and the Bonferroni / methodology
inconsistencies flagged by the severe reviewer have been
addressed in code and in the manuscript text. Two of the four
fatal flaws are textual fixes; one is a figure-construction fix;
one (the matched-budget question, F3) is now disclosed honestly
in a new caveats subsection rather than silently confounded with
the leakage-removal claim.

---

## 1. Reviewer-flagged fixes addressed

### 1.1 F1 — Long-only vs long/short mismatch

The paper text described a "long-only sign-following" strategy in
five places. The actual `src/backtest.py` implements a
sign-following \emph{long/short} strategy with positions in
$\{-1, +1\}$. All five textual occurrences were updated to
"long/short". The §10.6 (now §5.7) prose explicitly states:

> position $+1$ when the model predicts UP and $-1$ when it
> predicts DOWN, ... The position therefore takes a short
> exposure on every DOWN prediction; the strategy is not
> long-only.

The "no economic value" finding survives the relabel: the
realised net-equity curves were always those of a long/short
strategy; only the verbal description was wrong.

### 1.2 F2 — Synthetic vs real buy-and-hold path

`scripts/make_paper_figures.py` previously synthesised the B&H
curve as a constant compounded daily return anchored to the
metric file's `buy_and_hold_total_return`. The reviewer caught
that this construction smooths through the actual drawdowns
(notably the March 2020 crash and the 2022 tech selloff) and
visually flatters the underperformance gap.

The script now loads
`data/snapshots/AAPL_2020-01-01_2024-12-31.parquet` and
constructs the B&H curve from the real closing-price path,
normalised to start at 1.0 on the first OOF exit date in each
panel. The figure caption already carries the "Buy \& hold (real
path)" label, and the png at
`figures/backtest_equity_curves.png` was regenerated. The
qualitative finding stands: at every horizon, every model
strategy underperforms the real B&H curve. The h=15 race is
slightly closer than the synthetic version had suggested,
because the real path drops in 2022 before recovering.

### 1.3 F3 — Leakage gap confounded with under-tuning

The 22-percentage-point gap between the published-style
$0.730$ accuracy in §4 and the leakage-safe $0.513$ in §5 was
generated under different hyperparameter budgets:
50 Optuna trials and 5 inner folds in §4, 5 trials and 2 inner
folds in §5. A new subsection §5.5
(`sec:addendum_caveats`) explicitly discloses this confound and
names the matched-budget run (50 trials, 5 inner folds, all
five tree backends, leakage-safe protocol) as "the single
strongest robustness check we have not yet run". The §5.2
prose was also updated to acknowledge that the abbreviated
budget is a compute compromise and to refer the reader forward
to the caveats subsection for the implication.

### 1.4 F4 — n=1 sample claimed as "machine-learning stock-direction predictability"

The abstract was tightened from "We reassess machine-learning
stock-direction predictability" to "We reassess machine-learning
directional predictability \emph{on a single canonical large-cap
U.S.\ equity (AAPL, 2020-2024)}", with an explicit second
sentence stating that the panel-wide replication is gated on
multi-day compute and left for a follow-up. The introduction's
target sentence was tightened the same way and now explains why
AAPL was chosen (it is the most-studied stock in the published-
style directional-forecasting literature, with multiple
$\geq 70\%$ accuracy claims to anchor the reassessment to).

### 1.5 Bonferroni arithmetic error

The §5.5 caption previously claimed the Bonferroni adjustment
at $\alpha=0.05$ would retain "only the $h=5$ RF vs.\ XGBoost"
comparison. The smallest reported $p$-value in
Table~`tab:addendum_dm` is $0.007$, which exceeds the corrected
threshold $0.0556 / 9 \approx 0.0056$. The caption now states
explicitly that none of the nine pairwise comparisons survives
Bonferroni and that the reported model ranking should be read
as suggestive only. The accompanying prose was updated
similarly.

### 1.6 Methodology inconsistency on wavelet smoothing

Section §2.2 described wavelet smoothing as the label-construction
default without disclosing that the DWT filter is two-sided
(uses observations at $t' > t$ when reconstructing the denoised
series). A new \textbf{Wavelet smoothing is non-causal}
paragraph in §2.2 makes this explicit, points the reader at
`docs/audit/03_label_leakage_report.md`, and clarifies that the
wavelet-smoothed protocol is retained in §2-§4 because it
describes the originally-implemented pipeline whose accuracy
figures are reported in §4, while §5 replaces it with raw
realised returns and treats wavelet labels as a robustness
comparison only.

### 1.7 Data-availability statement overclaim

The Data Availability statement previously said the paper's
data was "committed as immutable Parquet snapshots", which is
true only for AAPL. The statement now distinguishes the
single-asset leakage-safe results (committed AAPL snapshot,
byte-identical reproducibility) from the published-style §4
results (full 25-ticker panel; snapshots not yet committed
because the panel-wide nested-CV sweep is gated on compute) and
points the reader at `scripts/snapshot_data.py` for on-demand
re-fetching from the same yfinance vintage.

## 2. Verification

Style scrub on the post-revision manuscript:

```
Missing labels        : none
Missing bib keys      : none
LaTeX em-dashes (---) : 0
Unicode em-dashes     : 0
'underscore' tokens   : 0
'economically meaningful' : 0
'long-only' mentions  : 1   (the explanatory "is not long-only")
```

Test suite: 80/80 pass on this branch (no code changes outside
`scripts/make_paper_figures.py`, which is not under test).

## 3. What this branch deliberately did NOT do

The reviewer's full list also includes:

- **Re-running the leakage-safe protocol with matched budget**
  (50 Optuna trials, 5 inner folds, all five tree backends).
  This is the actual answer to F3, not a textual hedge; the
  fix on this branch is to disclose the confound honestly.
  The user's chosen path (option 2 in the review handoff) was
  to disclose the limitation and submit, not to run the
  multi-day compute job.
- **Running the panel-wide nested-CV sweep across the other 24
  tickers.** Same reason. The abstract and conclusion now
  state the single-asset scope.
- **Romano-Wolf or Hansen-SPA correction of the DM matrix.**
  The new caption notes that no comparison survives a
  Bonferroni correction; a correlation-aware adjustment would
  be tighter, but it has not been computed.
- **Citing the 16 unused bibliography entries from the qf-07
  literature pass.** The new introduction already cites several
  of them; the remainder are reserved for a follow-up that
  expands the literature comparison.
- **Reconciling figure dimensions across the 14 legacy
  figures.** Already deferred from qf-09 and qf-10.

## 4. Status of the eleven-agent pipeline

| Branch | Status |
|---|---|
| qf-01 to qf-06 | Closed |
| qf-07 (literature) | Closed |
| qf-08 (QF format) | Closed |
| qf-09 (figures) | Closed |
| qf-10 (writing) | Closed |
| qf-11 (severe review) | Closed (verdict: Reject) |
| qf-12 (revision fixes) | **Closed on this branch** |

The revision fixes address the textual misstatements and the
fabricated-baseline issue called out by the severe review. The
matched-budget run, the panel sweep, and the Romano-Wolf
correction are now disclosed limitations rather than silent
confounds; whether to run them before submission is a separate
decision.

## 5. What the user must do before submission

1. Compile the manuscript locally with the QF template
   (`pdflatex` + `bibtex` cycle as documented in
   `13_qf_format_report.md` §6) and verify (i) the new §5.5
   caveats subsection renders, (ii) the regenerated
   `figures/backtest_equity_curves.png` displays the real
   AAPL price path, and (iii) the four end-statements still
   appear in the canonical order.
2. Read the new abstract and conclusion alongside co-authors
   and confirm framing.
3. Decide whether to run the matched-budget leakage-safe
   protocol before submission. If yes, the §5.5 caveats
   subsection should be revised once results land. If no, the
   limitation disclosure on this branch is sufficient to pass
   a real referee on grounds of intellectual honesty.
4. Decide whether to commit the full 25-snapshot panel before
   submission. The single command is
   `python scripts/snapshot_data.py`; it is a one-to-two
   minute operation and shrinks the data-availability
   statement to "all 25 snapshots are committed". This is the
   cheapest of the deferred items.
