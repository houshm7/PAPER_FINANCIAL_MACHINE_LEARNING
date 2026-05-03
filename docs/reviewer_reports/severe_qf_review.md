# Severe Referee Report

**Recommendation:** Reject (with explicit invitation to a fundamentally
restructured resubmission).

This report is harsh by request. The paper has clear methodological merit
on the leakage point, but in its current form it overreaches the evidence,
mislabels its own backtest, presents internally inconsistent methodology,
and rests its main empirical claim on a single asset with a chronically
under-tuned harness. None of these issues is repairable by minor revision.

---

## 1. Fatal flaws

**F1. The backtest in Section 5.7 is not the strategy the paper describes.**
The abstract (`final_paper/main.tex` lines 66-69), the introduction
(lines 142-147), the conclusion (lines 947-950), and the §5.7 narrative
(lines 1322-1334) all state that the backtest is a *long-only*
sign-following strategy. The implementation in `src/backtest.py` lines
180-208 sets `df["position"] = df["y_pred"]` where `y_pred ∈ {-1, +1}`
(see `src/pipeline.py` line 296: `y_pred_signed = (y_pred_bin * 2 - 1)`).
The realised positions are therefore long *and* short. This is confirmed
by the committed metrics file `results/backtest_metrics.csv`: at h=1,
Random Forest has `n_long=670` and `n_short=559` out of 1,229 trades; at
h=15, XGBoost has 61 long and 20 short. The audit document
`docs/audit/08_backtest_report.md` line 37-39 also describes the strategy
as long-short. The paper's headline economic claim ("no model-horizon
combination beats passive buy-and-hold once realistic costs are imposed")
is therefore based on a long/short strategy that the paper text never
describes. A long-only restriction would behave very differently in a
market with a 26% annualised drift, since the short leg drags during a
multi-year bull run. The paper either mis-describes its strategy or
mis-implements it; either way the §5.7 economic claim is not credible
until reconciled.

**F2. The buy-and-hold benchmark in Figure 12 is synthetic, not realised.**
`scripts/make_paper_figures.py` lines 78-103 anchors B&H to a *constant*
compounded daily return derived from the endpoint total return:
`daily = (1 + bh_total)^(1/n) - 1`, then `bh_curve = (1 + daily)^t`. This
draws a smooth exponential curve from 1.0 to the realised endpoint. The
real AAPL price path includes a 30% drawdown in 2022 and recovery in 2023.
Plotting a smooth synthetic curve against the strategies' jagged
realisations visually flatters or punishes the strategies in periods of
B&H volatility. It is also the only B&H curve the reader sees. The
manuscript's "no economic value" verdict (Conclusion line 952-955) is
anchored to this figure, so the verdict is anchored to a benchmark that
is not the actual benchmark. Replace with the realised
`prices.loc[bh_start:bh_end].pct_change().cumprod()` curve; do not
report the constructed proxy.

**F3. The headline 22-point leakage gap is computed under a deliberately
under-tuned harness, and the paper conflates "under-tuning" with
"leakage."** The leakage-safe pipeline runs *5 Optuna trials per
(outer fold, model)* (`results/nested_cv_run_snapshot.json`,
`n_trials: 5`) over a search space with 5-7 hyperparameters per model
(see `src/tuning.py` lines 33-113); the published-style replication uses
50 trials with the same TPE sampler (manuscript line 614). The §5.3
Table `tab:addendum_aapl_h1` rejects the legacy 0.73 at z = -17. Any
reader reasonably wonders how much of the 22-point drop reflects
leakage and how much reflects 5 vs. 50 trials and 2 vs. 5 inner folds.
The paper does not run that decomposition. The audit document
`docs/audit/06_nested_cv_report.md` and the §5.3 caption (line 1075)
acknowledge the trial budget is reduced "because the full sweep is
multi-hour per asset." That is fine as a compute statement; it is not
fine as a basis for the headline scientific claim. To attribute the
gap to leakage rather than trial budget, the authors must run at
minimum a same-budget control: the published-style pipeline at 5
trials, or the leakage-safe pipeline at 50. Neither is reported.

**F4. The paper draws conclusions about machine-learning predictability
of stock direction from n=1.** The abstract (line 45) states that the
paper "reassess[es] machine-learning stock-direction predictability".
Sections 3-4 build a 25-stock, 5-sector universe; Section 5 reports
results on AAPL only. The "leakage-safe" verdict, the "no economic
value" verdict, and the rejection of the 73% number all rely on this
single asset. AAPL's 2020-2024 sample is a textbook example of the
asset for which the literature claims work: high liquidity, long bull
run, dominant momentum. The paper itself notes (line 1365-1374) that
the panel sweep is "gated on compute". A single-asset reassessment is
a case study, not a reassessment of a literature. The framing of the
abstract and conclusion (line 957-966) overreaches what the data
support. Either reframe the contribution as a case study, or run the
panel.

---

## 2. Major comments

**M1. Asymmetric inference between §4 and §5.** The published-style §4
reports point estimates without confidence intervals at all
(Tables 4-7); §5 supplies block-bootstrap CIs only for the leakage-safe
estimates. This asymmetry biases the comparison: the §4 numbers look
deceptively precise. If the §4 73% had a Politis-Romano block-bootstrap
CI under the published-style harness, the rejection in
Table `tab:addendum_aapl_h1` at z = -17 would be against a CI, not a
point estimate. Apply the same inference machinery on both sides of
the comparison.

**M2. Asymmetric model coverage.** The abstract (line 64-65) says
"five tree-based classifiers" and Table 5 lists all five. §5 drops
CatBoost and Gradient Boosting (line 1073: "three tree backends out of
the original five"). The text justification is "compute". This is
unsatisfactory: CatBoost was the §4 AUC leader (line 627, AUC = 0.770)
and would be the natural alternate hypothesis for any reader. The
leakage-safe verdict on a model class that does not include the §4
leader is incomplete.

**M3. Inner CV with 2 folds is not a tuning signal.** `n_inner_splits=2`
(`results/nested_cv_run_snapshot.json`) divides each outer training
slice into two halves. With Purged K-Fold + 1% embargo on 800-1000
training rows, that yields ~400-500 row inner-train / inner-test halves
on noisy daily directional data. Two-fold tuning has too little
signal-to-noise to pick reliably between candidate hyperparameter
configurations; the per-fold drift in
Table `tab:addendum_drift` (line 1273-1287) is consistent with this.
The drift is then reported as evidence that legacy point estimates
"should be interpreted as the result of one such draw, not as 'the'
tuned hyperparameters" (line 1290-1294), but the same drift is
plausibly an under-powered tuner producing noise.

**M4. Diebold-Mariano p-values without multiple-testing correction
underwrite §5's only positive claim.** §5.5 reports nine pairwise DM
tests in Table `tab:addendum_dm` (line 1228-1244) and concludes that
"Random Forest dominates at h ≤ 5 [...] and XGBoost takes the lead at
h=15" (line 1246-1250). The caption on line 1224-1226 acknowledges
that nine tests at α=0.05 require Bonferroni p<0.0056 and that under
that threshold only RF vs XGB at h=5 survives. The §5.5 narrative
text then says the ranking is "real" anyway. That contradicts the
caption. With multiple-testing correction the paper has one
significant pairwise comparison, not a model ranking. The paper's
own §5.5 limitations (line 1258-1263) call for Romano-Wolf or
Hansen-SPA. Run those *before* the model-ranking claim, not as
"appropriate next step".

**M5. Section 2 (Methodology) describes a different pipeline from
Section 5 (the leakage-safe analysis).** §2.2 (line 226) constructs
labels from `\tilde{P}_{t+h} > \tilde{P}_t` (wavelet-smoothed); §5.3
(line 1067-1070) drops wavelet labels because the implementation is
non-causal; `src/preprocessing.py` lines 13-19 confirm the wavelet
path uses a global `pywt.wavedec` and is acknowledged non-causal.
§2.3 in line 329 sets K=5 with 1% embargo "following López de Prado"
without ever stating that this single-loop K-fold is *exactly the
leakage path the paper calls out*. §2.4 cites SHAP as a primary
interpretability device; §5 reports no leakage-safe SHAP. The
methodology section in its current form documents the *legacy*
pipeline and is internally inconsistent with §5. Rewrite §2 around
the leakage-safe pipeline; demote the wavelet/single-loop description
to a "what the prior literature does" subsection.

**M6. Wavelet smoothing is the methodological centrepiece of §2-§3 but
the leakage-safe addendum drops it.** §3.2 line 351-355 makes the
smoothing comparison the dominant figure of §3 (Figure 1). §3.3 line
369-371 describes feature selection on labels constructed from
smoothed prices. §5.3 line 1067-1070 says the wavelet smoothing is
non-causal. So Figure 1, the §3.2 narrative, and the entire §2.2
"Label smoothing" subsection are advertising a property of a labeling
scheme the paper itself disowns by §5. The reader is asked to absorb
substantial methodological exposition for an option that the
authors then flag as leakage. Keep the smoothing figure as a
*counter*-example, or remove it.

**M7. Buy-and-hold of 26.5%/yr 2020-2024 is a sample-period artefact.**
The conclusion's "no economic value" claim (line 957-966) implicitly
assumes B&H is a meaningful benchmark. The 2020-2024 window is
historically among the best 5-year windows for AAPL: it includes the
COVID rebound, the AI rally of 2023-2024, and only one moderate
2022 drawdown. Across other 5-year windows AAPL B&H is materially
lower. The paper's "no strategy beats B&H" claim is therefore
period-specific. Either run a robustness window (2008-2012 included),
or restate the claim as "over this sample window."

**M8. The hyperparameter drift table proves nothing on its own.**
Table `tab:addendum_drift` shows a 5x swing in `learning_rate` and
`n_estimators` across 5 outer folds. With 5 trials per fold and 2
inner folds the swing is the *expected* behaviour of an under-powered
tuner: TPE's 5-trial sample of a 7-dimensional space is essentially
random. The text (line 1289-1294) interprets the drift as evidence
that legacy single-fold estimates are unreliable. That is a possible
reading; an equally consistent reading is that 5 Optuna trials
constitute under-tuning, full stop. Without a same-budget comparison
(F3 above), the table is decorative.

**M9. Section 5 reports h=15 as "tied with chance" but also reports
56% pooled accuracy, AUC 0.525, F1 0.685.** The §5.4 paragraph
(line 1170-1182) admits the elevated accuracy "partly reflects the
model successfully exploiting class imbalance via majority-class
prediction rather than a genuine directional signal." A reader who
trusts that paragraph cannot also trust the §5.5 DM-based claim
that "XGBoost takes the lead at h=15" — the lead is between
class-imbalance exploitations. Reconcile: either the h=15 figure
is signal (and DM is informative), or it is class imbalance (and
DM is not interpretable as "directional skill"). The paper currently
asserts both.

**M10. Reproducibility claim is overstated for the panel.** Line 1417
("Data availability statement") says daily OHLCV for 25 equities is
committed under `data/snapshots/`. The actual snapshot directory
contains exactly one parquet (`AAPL_2020-01-01_2024-12-31.parquet`)
plus metadata. The other 24 tickers used in §3-4 are *not* committed.
A reader wanting to reproduce §4 would need to refetch from yfinance,
which the paper itself acknowledges varies. Match the data-availability
statement to the actual snapshot scope.

**M11. The §1 "70 to 95 percent" framing is uncited.** Line 91-93
states "Reported directional accuracies in the applied literature
commonly fall between 70 and 95 percent for tree-based and deep
models on individual equities, including high-volume names such as
AAPL." This is the headline framing for the entire paper. It is not
cited. The cited surveys (`sezer2020financial,jiang2021applications,
htun2023survey`) on line 101 are general; they do not document this
specific range. Add the citations or soften the claim.

---

## 3. Minor comments

**m1.** Figure 1 file is named `figures/smoothing_comparaison.png`
(French spelling, line 359). Keep the file name internal but the user-
facing references are fine; this is a hygiene issue.

**m2.** §2.4 caption for Figure `fig:pipeline` (line 496) advertises
"50 trials per model" for tuning. The leakage-safe §5 actually uses 5.
Either annotate the figure with both budgets or drop the trial count
from the caption.

**m3.** Table `tab:hyperparameters` (line 633-650) reports a single
"selected" hyperparameter row per model. By the paper's own §5.6 logic
this is a draw, not a population estimate. Either delete the table or
annotate it explicitly as a single-fold realisation.

**m4.** Line 1379 ("balanced-accuracy and Brier metrics, which would
clarify whether the h=15 point estimate is partly a class-imbalance
artefact, are deferred to a metric-upgrade revision") concedes that
the h=15 result needs a metric the paper does not provide. The metric
takes one line of code on the existing OOF predictions; computing it
would tighten a footnote-level claim into a primary one. Do it.

**m5.** §5.6 "What survives" (line 1296-1304) leaves the reader with
the impression that §4.1 (Table 3) is methodologically clean. §4.1 is
clean *given* the legacy single-loop pipeline, but Table 3 reports
67-84% Purged K-Fold accuracies that the §5 leakage-safe analysis
implies are still inflated. Make explicit that Table 3 inflates by L1
and L2 too, just less than its standard K-Fold counterpart.

**m6.** The "Use of generative AI" disclosure (line 1390-1406) says
Claude "was used to draft and refactor Python implementations of the
nested Purged K-Fold cross-validation harness, the Politis-Romano
stationary block-bootstrap routine, and ancillary plotting and
configuration utilities. [...] Claude was not used to [...] write the
substantive analytical text of the manuscript." The §5 prose is
clearly *in part* AI-drafted (clean parallel constructions, neat
"three findings" enumerations); the audit folder
`docs/audit/15_writing_report.md` documents writing assistance. The
disclosure is technically defensible but reads as understating
involvement. Consider rewording.

**m7.** The bibliography contains 47 entries
(`final_paper/references.bib`); the `\cite{}` graph uses 31. Sixteen
entries are not used: `avramov2023machine`, `bailey2014deflated`,
`bailey2014pseudomath`, `bussmann2021explainable`,
`demiguel2020transaction`, `fama1970efficient`, `guyon2003introduction`,
`harvey2016cross`, `huang2005forecasting`, `kelly2023financial`,
`kim2003financial`, `krauss2017deep`, `lundberg2020local`,
`patel2015predicting`, `sirignano2019universal`, `white2000reality`.
A bibliography that lists references the author never invokes is a
red flag; it suggests citation padding.

**m8.** §5.5 line 1227 references `Bonferroni-style adjustment at
α=0.05 would require p<0.0056 and would retain only the h=5 RF vs
XGB comparison`. The paper's own raw DM file
`results/inference_dm_pvalues.csv` shows that comparison's p-value
as 0.00736. That is *above* 0.0056. Bonferroni would retain *zero*
comparisons, not one. Recheck the arithmetic.

**m9.** Line 1093 in `tab:addendum_aapl_h1` lists pooled OOF accuracy
of 51.3% for RF; the inference file `inference_oof_cis.csv` reports
51.34%. Round consistently.

**m10.** §3.2 line 343 documents the panel as 25 stocks across five
sectors. The committed snapshot directory has one. Update the data
section to reflect what is actually committed for §5.

---

## 4. Unsupported claims

**U1.** Abstract line 47-49: "evaluate whether remaining predictive
signals carry economic value once realistic trading frictions are
imposed." The realised evaluation is *one* cost level (10 bps), *one*
strategy class (sign), *one* asset (AAPL), *one* sample window
(2020-2024). The plural ("frictions") and the singular framing of
"economic value" both overstate the evidence base.

**U2.** Conclusion line 957-966: "no examined trading rule beats a
passive benchmark net of costs over the same out-of-fold window."
The "examined trading rule" set is sign-following long/short at
fixed cost. The "passive benchmark" is a synthetic constant-daily-
return curve (F2). Restate with both qualifications.

**U3.** Line 970-973 ("The empirical evidence is single-asset; the
panel-wide nested-CV sweep across the other 24 tickers is gated on
compute and is left for a follow-up.") is the right disclosure but it
appears in the *Limitations* paragraph, after the Conclusion's
headline claims. The single-asset limitation should appear in the
abstract and the headline of §5.

**U4.** Line 169-171: "Our framework has three components: (i)
construction of predictive features from lagged price data, (ii)
definition of a directional target variable using a noise-reduced
price series, and (iii) out-of-sample evaluation via Purged K-Fold
cross-validation." This describes the legacy pipeline. The actual
research framework, per the §5 addendum, is the *nested* purged
pipeline with raw labels. The introduction's framework summary is
the wrong pipeline.

**U5.** Line 754: "CatBoost leads in three of five sectors,
plausibly because its ordered boosting and built-in target encoding
provide stronger regularization in the noisier settings." CatBoost
target encoding is irrelevant when there are no categorical features
in the feature set (the 13 selected features are all numerical
technical indicators). Drop this claim or substitute a plausible
mechanism.

**U6.** Line 1297-1304: "The core methodological contribution of the
paper, that overlapping-label leakage materially inflates standard
K-fold accuracy and that purged validation is the appropriate fix,
survives in full." This methodological claim *is not original*. It is
exactly Lopez de Prado (2018), Bailey & Lopez de Prado (2017), and
Bergmeir et al. (2018). The paper's contribution is *empirical*, not
methodological — that single-loop selection/tuning is also leaky on
this asset, despite using "purged" K-Fold. Restate.

---

## 5. Formatting issues

**fmt1.** Line 359 figure file name `smoothing_comparaison.png` (French
spelling) — minor.

**fmt2.** Tables 4-7 use `\scriptsize`; the matching tables in §5 also
use `\scriptsize`. The formatting is consistent within the paper but
borderline unreadable in a final manuscript.

**fmt3.** §2.3 ("Cross-validation, overlapping labels, and information
leakage") is heavy on math and rederives standard López-de-Prado
results (lines 240-329). For a Quant Finance paper this is too much
exposition of a known framework. Compress to half a page and cite.

**fmt4.** §1 introduction is split into four `\\` paragraphs that read
as a Claude-style enumeration ("First, [...] Second, [...] Third,
[...]" line 97-112). Tighten.

**fmt5.** §5 is labelled `Leakage-Safe Replication under Nested Purged
K-Fold` and presented as an "addendum" (line 998). For a final
submission, "addendum" framing is wrong: this *is* the main result.
Restructure: §5 becomes the body, §4 becomes a short
"published-style baseline" subsection inside §5 or a robustness
appendix.

**fmt6.** Two-column or one-column? The class is `rQUF2e`. No layout
issues observed beyond the table density above.

**fmt7.** Line 1011 "supersedes the headline accuracy figures in the
abstract, Tables~4--7, and the second paragraph of the Conclusion" is
helpful for the audit trail but is unusual prose for a submission. A
QF reader will not navigate the addendum-supersedes-abstract logic.
Rewrite §5's introduction to read as the paper's main results section.

**fmt8.** Many em-dash-shaped phrases use `--` (LaTeX en-dash) where
the project's CLAUDE.md rules prohibit double hyphens in prose. The
double hyphen renders as en-dash in PDF. Examples: lines 145, 158,
161-162, etc. Cosmetic.

---

## 6. Novelty assessment

**Methodological novelty:** Low. Nested Purged K-Fold with
per-fold feature selection and Optuna tuning is a direct application
of standard practice (López de Prado 2018 ch. 7; Cawley & Talbot 2010
on nested CV; Kapoor & Narayanan 2023 on leakage). The
implementation is competent but does not contribute new theory.

**Empirical novelty:** Modest. The 73 → 51% gap on AAPL is
striking and worth reporting, *if* the under-tuning confound (F3)
can be ruled out. The block-bootstrap inference at h=15 is a
useful demonstration that iid CIs are wrong on overlapping labels.

**As-submitted novelty:** Insufficient for QF. The paper is one
asset, one window, one strategy, one cost. The adjacent literature
(Bailey-Lopez de Prado 2014 deflated Sharpe, Harvey-Liu-Zhu 2016
Lucky Factors, Avramov et al. 2023 on ML in asset pricing) sets a
much higher bar for empirical claims.

**Verdict:** Single-asset case study presented as a literature
reassessment. Reframe as a case study or run the panel.

---

## 7. Economic relevance assessment

**Strategy menu is too thin.** Sign-following only. No
volatility-targeted strategy, no Kelly sizing, no long-only variant
that matches the paper's text (F1), no transaction-cost-aware
position sizing, no slippage model. For a Quantitative Finance
audience the strategy menu is the deliverable.

**Cost grid is too thin.** A single 10 bps level is reported in the
figure. The metrics CSV contains 0/5/10/25 bps but the paper does
not report the sensitivity. A QF referee wants to see the cost
break-even.

**Benchmark is wrong.** F2: synthetic compounded curve, not the
realised B&H path. A QF reader cannot judge drawdown timing.

**B&H is sample-period favourable.** M7.

**Sharpe of B&H is 0.90** (in `results/backtest_metrics.csv`).
The §5.7 narrative says "annualised return 26.5%" but does not
report Sharpe; the 0.90 number is itself sample-favourable.

**Net of costs the leakage-safe RF h=1 strategy returns -8% per
year at 10 bps** (`results/backtest_metrics.csv`,
`net_ann_return = -0.0809`). The paper says it underperforms B&H.
It does so by about 35 percentage points per year. State that.

**Verdict:** The economic-value claim is directionally correct but
under-reported. Expand the strategy and cost sweeps; report a
single summary table; replace the figure benchmark.

---

## 8. Reproducibility assessment

**What works.** The pipeline contract is encoded in `src/pipeline.py`
and unit-tested in `tests/test_nested_pipeline.py` (claimed 80 tests;
not run by this referee). The block-bootstrap and DM modules
(`src/inference.py`) have explicit seeds. Run snapshots
(`results/nested_cv_run_snapshot.json`,
`results/backtest_run_snapshot.json`) record settings. The committed
AAPL parquet snapshot is byte-pinning the inputs.

**What does not.** The data-availability statement (line 1417-1427)
claims 25 stocks are committed; only AAPL is. The §4 published-style
results in Tables 3-9 cannot be byte-reproduced without re-pulling
yfinance for the other 24 tickers.

**Code/manuscript drift.** §2.4 Figure 1 declares 50 Optuna trials;
§5 actually uses 5. The pipeline schematic does not match the
implementation.

**AI-drafted code.** The "Use of generative AI" disclosure is short
and concedes the harness was AI-drafted. The audit reports
(`docs/audit/01_*.md` to `15_*.md`) document much more extensive
AI involvement than the manuscript discloses.

**Verdict:** Single-asset reproducibility is solid. Panel
reproducibility is asserted but not delivered.

---

## 9. AI-tone assessment

The §1-§3 prose has the unmistakable signature of LLM drafting:
parallel "First, [...] Second, [...] Third, [...]" enumerations
(lines 97-112), neat "three findings emerge" frames (line 911), and
clean transition sentences ("Several findings merit discussion" line
887; "A striking finding is that" line 822-823). The §5 prose
is somewhat tighter but retains the cadence. The phrase "underscore"
is absent (good; the project's CLAUDE.md flags it). The phrase
"This finding is consistent with" appears multiple times. Em-dashes
are absent in prose (good).

The footnote-style "Note added in revision" pattern in §5 is
unusual for a final submission and reads as the trace of an
agent-driven revision rather than a single authorial hand.

**Severity:** Moderate. The prose is professionally clean and not
obviously machine-generated, but the structural cadence is
recognisable. A pass by a human author would smooth it. The AI
disclosure (line 1390-1406) is technically truthful but understates
the involvement that the audit trail in `docs/audit/` exposes.

---

## 10. Summary verdict

**Recommendation: Reject.**

The paper has a defensible empirical observation: on AAPL over
2020-2024, when feature selection and hyperparameter tuning are
quarantined inside the outer training slice, the headline 73% one-day
directional accuracy collapses to 51%. That observation is worth
publishing in some form.

In its current form, however, the paper:

1. Mis-describes its own backtest strategy (long-only vs
   long-short, F1).
2. Anchors its economic verdict to a synthetic benchmark curve, not
   the realised buy-and-hold path (F2).
3. Confounds the leakage gap with an under-tuning gap by running the
   leakage-safe pipeline at one-tenth the published-style trial budget
   (F3).
4. Frames a single-asset case study as a literature reassessment (F4).
5. Retains a methodology section that describes the legacy pipeline
   the §5 addendum supersedes (M5).
6. Reports asymmetric inference between §4 and §5 (M1).
7. Drops two of five tree-based models from the leakage-safe analysis
   without methodological justification (M2).
8. Bases its model-ranking claim on uncorrected pairwise DM tests
   that vanish under multiple-testing correction (M4) — and contains
   an arithmetic error in its own Bonferroni footnote (m8).
9. Inflates its bibliography by ~50% with uncited references (m7).

None of these is fixable by a minor or major revision in place. The
paper needs to be reorganised so that §5 becomes the body, §2 is
rewritten around the actual research framework, the backtest is
re-implemented and re-described, the benchmark figure is regenerated
on real prices, and at minimum a same-budget control run is added to
disentangle leakage from under-tuning. After those changes the paper
could be a competent single-asset case study with clear methodological
framing. A "reassessment of machine-learning stock-direction
predictability" requires the panel sweep that is currently gated.

The single-asset evidence, on its own, would support a short
methodological note (e.g. *Quantitative Finance Letters*) but not a
full QF article. As submitted: Reject.
