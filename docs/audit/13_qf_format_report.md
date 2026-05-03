# QF Format Pass — qf-08-qf-format

**Branch:** `qf-08-qf-format`
**Agent:** `.claude/agents/08_qf_format_agent.md`
**Status:** documentclass ported to `rQUF2e`, four required
end-statements + AI-use disclosure added, citation style
switched to `rQUF`, em-dashes and double-hyphens scrubbed from
prose, addendum tables now cited from text. The compile-test
itself is **not** done on this machine (no LaTeX toolchain
available); see §6.

---

## 1. What changed in `final_paper/main.tex`

### 1.1 Preamble and documentclass

```latex
% Old
\documentclass[12pt]{article}
... \usepackage{geometry}
... \usepackage{setspace}
... \geometry{margin=0.84in}
... \setstretch{1.15}

% New
\RequirePackage[2018-12-01]{latexrelease}
\documentclass{rQUF2e}
... (geometry / setspace dropped)
... + \usepackage{epstopdf}
```

`rQUF2e.cls` and `rQUF.bst` ship with the repository under
`journal_qf_template/`; the user's TeX build path needs to
include that directory (or copy the two files into
`final_paper/`).

### 1.2 Title block

The original `\title` / `\author` / `\thanks` / `\and` /
`\footnotemark[1]` block was replaced with the QF-canonical
form taken from `journal_qf_template/rQUFguide.tex`:

```latex
\title{Forecasting Stock Market Direction with Tree-Based Classifiers:
Evidence from Purged Cross-Validation and SHAP Interpretability}

\author{
H. MAJED$^{\ast}$$\dag$\thanks{$^{\ast}$Corresponding author. Email:
\texttt{houssem.majed@etu.univ-paris1.fr}.},
A. EL GHOUL$\dag$, H. TALBI$\dag$ and R. HENTATI KAFFEL$\dag$\\
\affil{$\dag$Universit\'e Paris 1 Panth\'eon-Sorbonne, \'Ecole d'\'Economie
de la Sorbonne, France}
\received{\today}
}
```

The corresponding-author flag is `houssem.majed@etu.univ-paris1.fr`.
Update the email or the author order before submission if
needed.

### 1.3 Keywords and JEL

The legacy `\noindent \textbf{Keywords:}` and `\noindent
\textbf{JEL Classification:}` paragraphs were converted to the
QF macros `\begin{keywords}...\end{keywords}` and
`\begin{classcode}...\end{classcode}`. The QF guide caps the
keywords list at 6; the previous draft had 7, so
``ensemble methods'' was dropped (it is a superset of
``tree-based classifiers'' and so is redundant).

### 1.4 Bibliography style

```latex
\bibliographystyle{apalike}   % old
\bibliographystyle{rQUF}      % new
```

The `rQUF.bst` style file is the journal-specific Harvard /
author-date style ships under `journal_qf_template/`. Citation
keys throughout the document remain unchanged; only the
rendered bibliography format differs.

### 1.5 End-statements (new, immediately before bibliography)

Four required Taylor and Francis statements were added in this
order, as `\section*{...}` (unnumbered) blocks per the QF
template's section 4.7 / 4.8 conventions and current Taylor
and Francis publisher policy (cf.
`docs/audit/12_literature_web_report.md`):

1. **Use of generative AI** (Methods-adjacent placement, just
   before Disclosure, per Taylor and Francis 2025 AI policy).
   Lists the Claude versions used, the specific scope (coding
   assistant for the empirical pipeline), and reaffirms author
   accountability.
2. **Disclosure statement.** "The authors report no conflict
   of interest." Update if any author has a relevant
   affiliation.
3. **Funding.** "This research did not receive any specific
   grant... ." Update if any author received targeted funding.
4. **Data availability statement.** Points to the committed
   Parquet snapshots under `data/snapshots/` and explains the
   vintage metadata.
5. **Code availability statement.** Points to the public
   GitHub repository
   <https://github.com/houshm7/PAPER_FINANCIAL_MACHINE_LEARNING>
   and pins the release tag `v0.2-leakage-safe-replication`.

### 1.6 Em-dashes and double-hyphens

Forty instances of `---` (LaTeX em-dash) and one Unicode em-dash
(U+2014) were scrubbed from prose. Replacement strategy:

- General rule: `(?<!-)---(?!-) ` → `, ` (comma + space). This
  preserves `----` sequences in TikZ comments (lines 328 / 338),
  which are the only places where four-dash sequences appear.
- Two label-pattern exceptions (`Stage 1, Correlation filter`
  → `Stage 1: Correlation filter` and the same for `Stage 2`)
  were corrected manually after the bulk pass because a colon
  reads more naturally than a comma at a label-style boundary.

The Unicode em-dash on line 56 (the abstract footnote) was
inside a parenthetical expansion of "leakage-safe nested
Purged K-Fold protocol" and read cleanly with `, `.

### 1.7 Addendum table citations

Three `\label{tab:addendum_*}` tables in §10 were defined but
never `\ref`'d from prose, which violates the Agent 08 rule
"Ensure all tables and figures are cited". Each was given an
explicit in-text reference:

| Label | Cited from |
|---|---|
| `tab:addendum_aapl_h1` | "The $z$-test in Table~\ref{...} rejects..." |
| `tab:addendum_aapl_window` | "...collapses under nested CV (Table~\ref{...})" |
| `tab:addendum_drift` | "are not stable across outer folds (Table~\ref{...})" |

## 2. What this commit deliberately does NOT change

The agent definition has 12 tasks. Tasks 1-10 are addressed
above. Tasks 11-12 are partially deferred:

- **Task 11 (all tables and figures cited).** The three
  addendum tables are now cited. Six legacy figure labels in
  the body remain orphans:
  `fig:individual_vs_portfolio`, `fig:portfolio_evolution`,
  `fig:sector_analysis`, `fig:shap_dep`, `fig:smoothing`,
  `fig:target_corr`. These are decorative in the current
  draft and have no inbound `\ref`. The agreed division of
  labour with `qf-09-figures-tables` is that figure curation
  (move to appendix, drop, or cite-properly) lives there, not
  here. This branch flags the orphans so qf-09 has an explicit
  to-do list.
- **Task 12 (move excessive figures to appendix).** Same
  reason: the trim-and-curate pass is the qf-09 scope.

The introduction itself was **not** rewritten on this branch.
The "These results underscore..." sentence in the abstract,
the unsupported "economically meaningful" parenthetical, and
similar Writing-Agent-flagged phrases all survived this pass
intact. They are the qf-10 scope.

## 3. Citation / cross-reference health

Verified by a one-shot Python check after the edits:

```
Missing labels (referenced but not defined): none
Missing bib keys (cited but not in bib)    : none
Orphan labels: 6 (all six legacy figures, deferred to qf-09)
```

The 22 unused bib keys (out of 47) are the new entries from
qf-07 plus a few legacy entries; they do not cause errors
under BibTeX (unused entries are silently skipped) and will
shrink as `qf-10-writing` weaves them into the rewritten
prose.

## 4. Style audit

- 0 LaTeX em-dashes (`---`) outside of TikZ comments.
- 0 Unicode em-dashes (`â`).
- 6 keywords (down from 7) per QF guide convention.
- All four required end-statements present, in canonical order
  (Disclosure, Funding, Data Availability, Code Availability),
  preceded by the AI-use disclosure.
- No claim of "economic value" was added on this branch (the
  pre-existing "economically meaningful" phrase in the legacy
  abstract is flagged for `qf-10-writing`).

## 5. Audit closure status after this commit

| ID | Status |
|---|---|
| C-1, C-2, C-3, C-4, C-5 | Closed before this branch |
| C-13, C-18, C-19, C-22 | Closed before this branch |
| Agent 07 (literature) | Closed on `qf-07-literature-web` |
| Agent 08 (QF format) | **Closed on this branch except the figure cleanup, which is queued for qf-09** |
| Agent 09 (figures) | Pending |
| Agent 10 (writing) | Pending |
| Agent 11 (severe review) | Pending (final step) |

## 6. What the user must verify locally

Because no LaTeX toolchain is installed in this session, the
following items are **not** machine-verified on this branch:

1. `pdflatex final_paper/main.tex` runs cleanly under
   `rQUF2e.cls`. The two highest-risk items are: (i) `hyperref`
   loaded after the `rQUF2e` class (it is loaded last in the
   preamble, which is the safer order); (ii) `subcaption`
   coexisting with anything `rQUF2e.cls` may itself load. If a
   `subfigure / subcaption` clash appears, drop `subcaption`
   and use `\usepackage{subfigure}` (the QF guide's
   recommendation).
2. `bibtex final_paper/main` resolves under `rQUF.bst`. The 21
   new bib entries from `qf-07-literature-web` use standard
   `@article` / `@book` / `@inproceedings` types; if `rQUF.bst`
   chokes on any specific field (DOI, URL), prune that field.
3. The four end-statements render in the order
   AI / Disclosure / Funding / Data / Code, before References.
4. The QF-specific numerics still flagged in
   `12_literature_web_report.md` §8 (word limit, abstract cap,
   keyword cap, etc.) are within bounds.

If pdflatex fails, the most likely root causes and fixes:

- "File `rQUF2e.cls' not found" → either copy
  `journal_qf_template/{rQUF2e.cls,rQUF.bst}` into
  `final_paper/`, or add `--include-directory=journal_qf_template`
  to the latexmk / pdflatex invocation.
- A `\received` overflow → wrap a long date in `{}` braces.
- A `keywords` environment error → ensure no blank line between
  `\maketitle` and `\begin{keywords}`.

## 7. Reproducing this run

```bash
git checkout qf-08-qf-format
# Inspect the changed sections
git diff qf-07-literature-web -- final_paper/main.tex | head -200
# Compile (requires rQUF2e.cls on TEXINPUTS)
cp journal_qf_template/{rQUF2e.cls,rQUF.bst,rQUF.sty} final_paper/ 2>/dev/null
cd final_paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The companion `12_literature_web_report.md` documents the
references this commit assumes are present in
`final_paper/references.bib`.

## 8. Audit closure

This commit closes Agent 08's scope as defined in
`.claude/agents/08_qf_format_agent.md` modulo the figure
curation (tasks 11-12), which is on the `qf-09-figures-tables`
branch by design.
