# Label-Construction Leakage Report

**Branch:** `qf-03-label-fixes`
**Audit cross-reference:** `docs/audit/01_repo_audit.md` issues C-1, C-5, C-7, C-10
**Status:** code fixed; experiments not yet rerun (per task constraints).

This report documents two leakage paths in the original label-construction
pipeline, the minimal code fix applied on this branch, and the unit-test
evidence that the fix holds.

---

## 1. Findings

### F-1. `create_target_labels` silently mapped unobserved horizons to DOWN [critical]

`src/preprocessing.py` (original):

```python
def create_target_labels(prices, window):
    future_price = prices.shift(-window)
    price_change = future_price - prices
    labels = np.where(price_change > 0, 1, -1)
    return pd.Series(labels, index=prices.index)
```

`prices.shift(-window)` produces `NaN` for the last `window` rows because
`P_{t+h}` is unobserved. NumPy's `NaN > 0` evaluates to `False`, so
`np.where` routed every one of those rows to `-1`. Effects:

- The last `window` rows of every label series carry a fabricated DOWN
  label rather than an explicit "unavailable" marker.
- Downstream code did not detect this — `prepare_features` then `dropna`-ed,
  but with no NaN in `target`, none of those rows were dropped.
- The class balance reported in the paper (~61/39 UP/DOWN) is mildly
  inflated toward DOWN by `window × n_stocks` synthetic DOWN labels.
- More importantly, every last-`window`-row in the panel is a label
  drawn from a degenerate "all DOWN" distribution unrelated to the
  underlying signal. Fold accuracy on those rows is uninformative.

### F-2. `prepare_features_with_t1` capped `t1` positionally [major; audit C-5]

`src/preprocessing.py` (original):

```python
end_positions = np.minimum(np.arange(len(X.index)) + window, len(X.index) - 1)
t1 = pd.Series(X.index[end_positions], index=X.index)
```

Two issues:

1. `t1` was indexed off the *cleaned* `X.index`, not the original
   `df.index`. Since `prepare_features` drops indicator warm-up rows but
   (due to F-1) **did not** drop the last `window` rows, the cleaned
   `X.index` was the same length as `df.index` minus the warm-up — so
   the boundary `np.minimum(..., len(X.index) - 1)` collapsed the last
   `window` `t1` values to a single timestamp.
2. The collapse made the López-de-Prado purge condition behave as if
   the last `window` observations all settled at the same calendar
   instant, causing the surrounding rows to be treated as overlapping
   when they should be considered terminal-and-dropped.

### F-3. Default label mode was wavelet, which is non-causal [critical; audit C-1]

`prepare_features(..., smoothing_method="wavelet")` was the project
default for label construction. The wavelet path uses `pywt.wavedec`
over the **entire** closing price series and a universal threshold
`σ √(2 log n)` estimated from the full sample's finest detail
coefficients. The smoothed price `\tilde{P}_t` therefore depends on
`{P_{t+1}, …, P_T}`, so the label

```
y_t = 1{ \tilde{P}_{t+h} > \tilde{P}_t }
```

is measurable with respect to the full future, not with respect to
information available at `t+h`. Purging on label horizon `h` cannot
remove this leakage path.

---

## 2. Fixes (code)

### Fix 2.1. `create_target_labels` returns explicit NaN at the boundary

```python
def create_target_labels(prices, window):
    future_price = prices.shift(-window)
    price_change = future_price - prices
    labels = pd.Series(np.nan, index=prices.index, dtype="float64")
    valid = price_change.notna()
    labels.loc[valid & (price_change > 0)] = 1.0
    labels.loc[valid & (price_change <= 0)] = -1.0
    return labels
```

- Last `window` rows: `NaN` (caller drops them via `dropna`).
- Tie semantics preserved (`price_change == 0 → -1`); changing tie
  handling is out of scope for this PR.

### Fix 2.2. `prepare_features` drops the trailing NaNs in one pass

`prepare_features` already calls `df_features.dropna()` after merging
indicators with the `target` column. With Fix 2.1 in place, that single
`dropna` now removes both the indicator warm-up rows *and* the last
`window` label-NaN rows. `X.index` and `y.index` are equal by
construction.

### Fix 2.3. `prepare_features_with_t1` builds `t1` from the original index

```python
full_index = pd.Index(df.index)
pos = full_index.get_indexer(X.index)
end_pos = pos + window  # never overflows because the trailing rows were dropped
t1 = pd.Series(full_index[end_pos], index=X.index)
```

- No `np.minimum` cap; if a position would overflow, that's a contract
  violation and we raise `RuntimeError`.
- `t1[i]` is a real calendar timestamp from `df.index`, exactly `window`
  trading-day rows after `X.index[i]`.

### Fix 2.4. New `label_mode` parameter; `raw_return` is the default

```python
LABEL_MODES = ["raw_return", "wavelet", "exponential", "savgol", "none"]
DEFAULT_LABEL_MODE = "raw_return"

def prepare_features(df, window, *, label_mode="raw_return",
                     smoothing_method=None, ...):
    ...
```

- `raw_return` (and its alias `none`): no smoothing, fully causal — the
  label uses raw `P_t` and raw `P_{t+h}`.
- `wavelet`, `exponential`, `savgol`: opt-in robustness modes. The
  module docstring flags `wavelet` and `savgol` as **non-causal** so
  callers know they are reporting leaky-by-design baselines.
- The legacy `smoothing_method=` kwarg is retained as a deprecated alias
  so the existing notebook does not break. When both arguments are
  supplied, `smoothing_method` wins (matches old behaviour).
- `prepare_features_with_t1` accepts the same parameters and forwards
  them.

---

## 3. Tests

`tests/test_label_construction.py` (21 cases, all passing) covers:

| # | Test | What it pins down |
|---|---|---|
| 1 | `test_create_target_labels_last_window_is_nan[1,2,5,10,15]` | Last `h` labels are NaN; earlier labels are observed; on a strictly increasing series every label is +1. |
| 2 | `test_create_target_labels_tie_semantics` | Constant-price ties remain DOWN (back-compat). |
| 3 | `test_prepare_features_drops_last_window_rows[1,5,15]` | The last `h` calendar dates do not appear in `X`; `y` has no NaN and only ±1. |
| 4 | `test_prepare_features_with_t1_index_alignment[1,2,5,10,15]` | `X.index == y.index == t1.index`; every `t1[i]` is `X.index[i] + h trading-day rows` in `df.index`. |
| 5 | `test_prepare_features_with_t1_no_boundary_collapse` | `t1` is strictly monotonic — no clustering at a single boundary timestamp. |
| 6 | `test_raw_labels_do_not_use_future_smoothing` | Two price series that agree on `[0, K]` and differ on `(K, N)` produce identical raw labels for every `t` with `t + h ≤ K`. **Causality property.** |
| 7 | `test_wavelet_labels_DO_use_future_smoothing` | Negative control: changing the tail of `Close` *does* change earlier wavelet labels. Documents the leakage signature; will fail (and should be revisited) if wavelet is ever made causal. |
| 8 | `test_label_mode_default_is_raw_return` | Default API contract. |
| 9 | `test_label_mode_none_equals_raw_return` | `"none"` and `"raw_return"` produce identical `X, y`. |
| 10 | `test_legacy_smoothing_method_alias_still_works` | Notebook calls that pass `smoothing_method="exponential"` continue to work. |
| 11 | `test_label_mode_unknown_value_raises` | `ValueError` on bad input. |

Run with:

```bash
python -m pytest tests/test_label_construction.py -v
```

`pytest` is not (yet) declared in `pyproject.toml`. Reproducers should
install it manually (`pip install pytest`) until a `dependency-groups`
or `[project.optional-dependencies]` block is added in a follow-up PR.

---

## 4. Backward-compat impact

- **API surface.** All previous call sites continue to work unchanged:
  `prepare_features(df, window)`, `prepare_features_with_t1(df, window)`,
  and any keyword call passing `smoothing_method=...` resolve to the
  same code path as before *except* that the trailing `window` NaN-label
  rows are now dropped instead of silently being mislabelled DOWN.
- **Default behaviour change.** A bare `prepare_features(df, window)`
  call now uses `label_mode="raw_return"` instead of `wavelet`. This is
  the intended fix per the task brief: raw returns are the only causal
  label construction available today and should be the headline-number
  baseline.
- **Notebook impact.** The committed `final_notebook/...ipynb` was *not*
  rerun (per task constraint). Its cell outputs continue to reflect
  the old wavelet defaults; rerunning will (a) drop a small number of
  trailing-boundary rows per ticker and (b) change all label values to
  raw returns. The downstream accuracy comparison must therefore be
  redone before any results-section text is updated.

---

## 5. What this PR does **not** change

Per the task constraints, this branch *does not* attempt to:

- Rerun the full `final_notebook/`.
- Edit `final_paper/main.tex`.
- Address the other criticals from the audit (C-2 nested CV, C-3 feature
  selection inside folds, C-13 data snapshot, C-19 economic backtest,
  C-22 inference). Those land on later branches.

The expected ordering remains the one set in `docs/audit/01_repo_audit.md`:
labels → data snapshot → nested CV refactor → raw-return ablation →
backtest → inference.

---

## 6. Headline takeaway for the paper

Once the notebook is rerun on this branch (not done yet), the paper's
headline numbers must be re-derived under raw-return labels. Two
empirical predictions to verify in the next branch:

1. The 11–33 pp Standard-vs-Purged gap should *narrow* somewhat once
   the leakage from non-causal wavelet smoothing is removed, because
   both validation schemes were partly inflated by the same source.
2. The single-asset accuracy at h=1 (currently reported as 73%) should
   move materially. Direction is unclear *a priori*: removing leakage
   pushes accuracy down, but removing the fabricated last-h DOWN labels
   pushes it slightly up. The net effect is an empirical question.
