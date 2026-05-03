# Writing Pass — qf-10-writing

**Branch:** `qf-10-writing`
**Agent:** `.claude/agents/10_writing_agent.md`
**Status:** abstract, introduction, conclusion, results-section
preface and §10 self-references rewritten under the leakage-safe
framing. Style scrub: zero em-dashes, zero double-hyphens in
prose, zero "underscore" tokens, zero "economically meaningful"
claims. Section §10 is no longer titled "Addendum"; it is the
substantive leakage-safe section.

---

## 1. What the abstract says now

The abstract is a clean 219-word paragraph that opens with the
target framing sentence verbatim:

> We reassess machine-learning stock-direction predictability
> under leakage-safe nested Purged K-Fold cross-validation and
> evaluate whether remaining predictive signals carry economic
> value once realistic trading frictions are imposed.

It then states the published-style replication (73~percent at
$h=1$), the leakage-safe figure (51~percent with the Politis-
Romano stationary block-bootstrap CI $[0.486, 0.541]$ and
$z=-17$ against 0.730), the multi-horizon walk-back
($\hat{\rho}_1 \approx 0.69$ at $h=15$ widens the CI to
$[0.481, 0.643]$, which contains chance, the legacy 0.532, and
the leakage-safe 0.560), the model ranking (RF at $h \le 5$,
XGB at $h=15$, LGB worst), and the economic finding (no
sign-following long-only strategy at 10~bps per round trip
exceeds passive buy-and-hold over the same OOF window). The
SHAP RSI dominance is retained as a one-sentence epilogue.

The pre-rewrite "Note added in revision" footnote and the
two trailing footnotes pointing to §10 have been removed: the
abstract IS the leakage-safe abstract now.

## 2. What changed in the body

| Section | Action |
|---|---|
| §1 Introduction | Replaced wholesale with the qf-07 drop-in (5 paragraphs, the leakage-safe framing). Reading-order pointers updated to point to §4 (published-style) and §5 (leakage-safe). |
| §2 Methodology | Untouched (it describes the data and pipeline, both compatible with both framings). Gained a `\label{sec:methodology}`. |
| §3 Data | Untouched. Gained `\label{sec:data}`. |
| §4 Results | Title changed from "Results" to "Results: Published-Style Replication under Standard Purged K-Fold". The first paragraph now hedges: it states explicitly that §4 reports the published-style analysis as a foil and that §5 supersedes the headline numbers; readers primarily interested in the leakage-safe answer are told they can skip directly to §5. Body unchanged. Gained `\label{sec:results}`. |
| §5 (was Addendum, §10) | Renamed from "Addendum: Leakage-Safe Replication under Nested Purged K-Fold" to "Leakage-Safe Replication under Nested Purged K-Fold". Eight prose self-references to "this addendum" / "the addendum" replaced with "this section" / "the leakage-safe section". Label keys (`sec:addendum`, `tab:addendum_*`, `fig:addendum_*`) preserved unchanged. |
| §6 Conclusion | Replaced wholesale. Three substantive findings under italicised headings: validation drives reported accuracy; honest inference reshapes the multi-horizon curve; no economic value net of costs. Limitations and future-directions paragraphs updated to reflect the actually-deferred follow-up items (panel-wide sweep, Romano-Wolf adjustment, balanced-accuracy / Brier metrics, walk-forward robustness). Gained `\label{sec:conclusion}`. |

Two prose-level fixes outside §1 / §5 / §6:

- §3.1 (line 202): "...nonlinear models yield economically
  meaningful gains in forecast accuracy" was replaced with
  "...measurable gains in forecast accuracy on cross-sectional
  asset-pricing tasks". The unqualified "economically
  meaningful" phrase was a Writing-Agent rule violation
  because the cited claim
  \citep{gu2020empirical} is about cross-sectional return
  prediction, not about the directional task we evaluate.
- §3 SHAP discussion (line 333): "This distinction is
  economically meaningful" softened to "This distinction has
  direct economic content".

## 3. Style scrub (Writing Agent rules 1-6)

A Python script verifies the post-rewrite state:

```
LaTeX em-dashes (---) in prose : 0
Unicode em-dashes (U+2014)     : 0
'underscore' tokens            : 0
'economically meaningful'      : 0
```

(The two `----` four-dash sequences inside TikZ source comments
in §3.4 are preserved; they are LaTeX comment delimiters, not
prose.)

Citation graph integrity is preserved:

```
Missing labels: none
Missing bib keys: none
```

## 4. List of claims that still require evidence

Per Writing Agent rule 10, the following claims appear in the
manuscript and are not yet fully supported by the empirical
artefacts on `main`. Each is flagged with the audit item that
would close it.

### Strong claims with full empirical backing

- **Leakage-safe AAPL $h=1$ accuracy is 51\%.** Backed by
  `results/nested_cv_predictions.csv` and `inference_oof_cis.csv`
  (qf-04, qf-06).
- **Leakage-safe $h=15$ is statistically tied with chance.**
  Backed by the same artefacts, with the block-bootstrap CI
  computed from the empirical lag-1 autocorrelation of the
  correctness series.
- **No backtested strategy exceeds buy-and-hold net of costs.**
  Backed by `results/backtest_metrics.csv`, the new
  `figures/backtest_equity_curves.png`, and qf-05 audit report.

### Claims that hold for AAPL but are extrapolated (gating: panel sweep)

- "How much of the published machine-learning advantage on a
  canonical stock disappears when validation is honest"
  (introduction). Currently demonstrated on a single stock.
  Generalisation requires the full $25 \times 5$ panel-wide
  nested-CV sweep that is gated on multi-day compute and
  documented as deferred in `07_nested_cv_first_run.md` §6.
  **If the panel sweep is not run before submission**, the
  introduction sentence should be tightened to "on AAPL".

### Claims that depend on multiple-testing corrections we have not yet implemented

- The pairwise Diebold-Mariano model rankings in
  Table~`tab:addendum_dm` (RF dominates at $h \le 5$, XGB at
  $h=15$). The nine reported $p$-values are uncorrected; a
  Romano-Wolf or Hansen-SPA adjustment is the appropriate next
  step, and was flagged as deferred in `09_inference_report.md`
  §5. The `qf-10` text says the ranking is "real" qualitatively
  but reminds the reader that, in absolute terms, all three
  models are statistically tied with chance at $h \ge 5$. If a
  reviewer pushes back, the corrected $p$-values should be
  computed and the table updated.

### Claims that depend on metrics not yet implemented

- The $h=15$ AUC of 0.525 with $f_1=0.685$ is described as
  "consistent with class-imbalance exploitation" rather than
  directional skill. This characterisation is plausible but
  not rigorously demonstrated; balanced-accuracy and Brier
  metrics, audit item C-26, would close the question. Until
  then, the conclusion's caveat ("Balanced-accuracy and
  Brier metrics... are deferred to a metric-upgrade revision")
  is the correct hedge.

### Claims that hold for the static outer-fold split only

- The 22-point leakage gap is reported under one outer-fold
  partition (the Purged K-Fold split with seed 42). Walk-
  forward robustness, audit item C-9, would test whether the
  finding survives an alternative time-aware partition. The
  conclusion's "future directions" paragraph names this
  explicitly.

### Style claims (no rewrite needed, but flagged for the severe reviewer)

- The introduction credits the Kapoor-Narayanan reproducibility
  finding to "seventeen affected fields" without an inline
  page citation. The reference is in the bib
  (`kapoor2023leakage`); a page-level inline citation can be
  added at proof stage if the reviewer asks.
- The conclusion's claim that the leakage-safe pipeline "combines
  per-fold per-asset feature selection and hyperparameter tuning
  with stationary block-bootstrap inference" is a
  contribution-statement, supported by the qf-04 and qf-06
  audit reports. If the reviewer asks for novelty positioning
  against \citet{lopez2018advances}, the relevant passage is
  in `06_nested_cv_report.md` §1.

## 5. What this branch deliberately did NOT do

Three Agent-09 deferred tasks (minimum-set figure curation,
appendix moves, dimension standardisation) were named in
`14_figures_tables_report.md` §4 as queued for `qf-10-writing`.
Of these, **only the dimension-and-curation pass was not
performed on this branch**. The reasons:

- The body §4 retains the legacy figure set because §4 IS the
  legacy archive in the new framing. Moving its figures to
  appendix would partially erase that archive role.
- The new §5 (leakage-safe) currently has one figure
  (`fig:addendum_backtest`); adding an appendix to host
  duplicates of the legacy figures alongside is redundant.
- Re-rendering legacy figures with bigger labels still requires
  the legacy notebook to run, which the standing constraint
  forbids.

These remain open for `qf-11-severe-review` to flag if the
reviewer pushes back.

## 6. Audit closure status after this commit

| Branch | Status |
|---|---|
| qf-01 to qf-06 (data and code) | Closed |
| qf-07 (literature) | Closed |
| qf-08 (QF format) | Closed |
| qf-09 (figures) | Closed (modulo curation, deferred) |
| qf-10 (writing) | **Closed on this branch** |
| qf-11 (severe review) | Pending (final step) |

## 7. What the user must verify before submission

The same `pdflatex` compile-test queued in
`13_qf_format_report.md` §6 still applies. In addition, the
user should:

1. Read the new abstract and conclusion end-to-end and decide
   whether the framing aligns with co-author preferences. The
   leakage-safe-as-headline framing is non-trivial and reframes
   the contribution; co-author concurrence is essential.
2. Decide whether to run the panel-wide nested-CV sweep before
   submission; if not, tighten the introduction to "AAPL" as
   noted in §4 above.
3. Decide whether to run the Romano-Wolf or Hansen-SPA
   correction before submission; if so, update Table
   `tab:addendum_dm` and the §5.5 prose accordingly.
