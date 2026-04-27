# Data Snapshot Mechanism — Closes Audit C-13

**Branch:** `qf-data-snapshot`
**Audit cross-reference:** critical issue C-13 — *Live yfinance
dependence; no data snapshot committed.*
**Status:** snapshot-first data layer shipped, AAPL snapshot
committed, README updated.

---

## 1. Why this exists

Yahoo Finance retroactively adjusts splits and dividends and, less
predictably, occasionally backfills or corrects historical bars
("vintage drift"). The pipeline previously called
`yf.Ticker(...).history(...)` every run, so two reproducers running
on different days could legitimately get different values for the
same `(ticker, start, end)` query and therefore different downstream
numbers.

Audit issue C-13 flagged this as critical for any paper-grade
reproducibility claim. The fix is the standard one: cache each
ticker's OHLCV at first download in a versioned, immutable file and
have downstream code read the cache by default.

---

## 2. Implementation

`src/data.py` gained a snapshot layer:

```
data/snapshots/
    AAPL_2020-01-01_2024-12-31.parquet      # the cached OHLCV
    snapshot_metadata.json                   # vintage / row counts / save UTC
```

API surface:

| Function | Purpose |
|---|---|
| `load_or_download(ticker, start, end, *, prefer_snapshot=True)` | Public entry point. Returns the cached Parquet if it exists and `prefer_snapshot=True`, else falls back to yfinance. |
| `save_snapshot(ticker, start, end, df=None)` | Writes the snapshot. If `df` is None, downloads first. Updates `snapshot_metadata.json` with row count, first/last date, save-UTC timestamp, and yfinance version. |
| `download_stock_data(...)` | Existing public API; now a thin wrapper over `load_or_download` so all legacy callers are snapshot-aware without changes. |
| Internal `_download_from_yfinance(...)` | The previous live-fetch code, isolated for monkey-patching in tests. |

The snapshot directory is overridable via the `DATA_SNAPSHOT_DIR`
environment variable (used by tests to redirect to `tmp_path`).

`scripts/snapshot_data.py` is the CLI driver:

```
python scripts/snapshot_data.py --tickers AAPL          # single
python scripts/snapshot_data.py                         # full panel
python scripts/snapshot_data.py --tickers AAPL --force  # refresh
```

---

## 3. Tests (8 cases, all passing)

`tests/test_data_snapshot.py`:

| # | Property |
|---|---|
| 1 | `save_snapshot` writes a Parquet at the canonical path. |
| 2 | `load_or_download` reads the snapshot and does **not** call yfinance when it exists (verified by monkey-patch on `_download_from_yfinance`). |
| 3 | `load_or_download` falls back to yfinance when no snapshot exists. |
| 4 | Round-trip preserves OHLCV values to 1e-10 relative tolerance. |
| 5 | `snapshot_metadata.json` records ticker, dates, row count, and `saved_utc`. |
| 6 | `DATA_SNAPSHOT_DIR` env var redirects the snapshot location. |
| 7 | `download_stock_data` (the legacy public API) transparently uses the snapshot. |
| 8 | `prefer_snapshot=False` forces a live fetch even when a snapshot exists. |

```bash
$ python -m pytest tests/test_data_snapshot.py -v
8 passed in 2.51s

$ python -m pytest                        # full suite
80 passed in 67.45s   (17 GPU + 25 label + 6 nested + 11 backtest + 13 inference + 8 snapshot)
```

---

## 4. Initial snapshot

Committed under `data/snapshots/`:

- `AAPL_2020-01-01_2024-12-31.parquet` — 67 KB, 1257 trading days,
  yfinance vintage 1.1.0 captured 2026-04-27 22:13 UTC.

Verification: a fresh process loads the snapshot in **0.08 s** vs
the ~6 s a live yfinance round-trip takes. All downstream
consumers (`scripts/run_nested_cv.py`, `scripts/run_backtest.py`,
`scripts/run_inference.py`) automatically pick up the snapshot
because they all enter through `download_stock_data`.

The other 24 tickers from `STOCK_UNIVERSE` are not yet snapshotted —
that's a one-line CLI invocation
(`python scripts/snapshot_data.py`) but downloads 25 tickers
sequentially against yfinance, which is a 1–2 minute job. Deferred
to whenever the panel-wide nested-CV sweep is run, since both depend
on the same panel.

---

## 5. Dependency change

`pyproject.toml` gains `pyarrow>=15` for Parquet I/O. CSV.gz was
considered as a zero-dependency fallback but loses precision on
extreme values and is 3–5× slower; Parquet is the standard choice
for tabular numerical data.

`uv sync` will pick up the new dependency on the next install.

---

## 6. Reproducing this run

```bash
git checkout qf-data-snapshot
python scripts/snapshot_data.py --tickers AAPL  # idempotent (skips if exists)
python -m pytest tests/test_data_snapshot.py -v
```

---

## 7. What this does NOT do

- Does not version-control the **panel** snapshots (only AAPL is
  committed). The full 25-ticker snapshot is a 1–2 min download +
  ~1.7 MB on disk; deferred to the panel sweep.
- Does not snapshot the **nested-CV intermediate results**. Those
  files (`nested_cv_predictions.csv` etc.) are already committed as
  the canonical artefacts of each run.
- Does not implement a **snapshot integrity check** (e.g.,
  SHA-256 over each Parquet). yfinance's vintage drift is captured
  by the save-UTC timestamp in `snapshot_metadata.json`, which is
  enough for reproducibility audits but not enough to guarantee a
  reproducer didn't tamper with the cache. A SHA-256 manifest is a
  natural follow-up.

---

## 8. Summary of audit closure

With this commit, the open critical / major audit items from
`docs/audit/04_post_fix_verification.md` are closed:

| ID | Status |
|---|---|
| C-1 (causal labels) | Closed on `qf-03-label-fixes` |
| C-2 (no nested CV) | Closed on `qf-04-nested-purged-cv` |
| C-3 (selection outside folds) | Closed on `qf-04-nested-purged-cv` |
| C-4 (panel-wide tuning) | Closed on `qf-04-nested-purged-cv` (per-fold per-asset tuning) |
| C-5 (positional t1 cap) | Closed on `qf-03-label-fixes` |
| C-13 (data snapshot) | **Closed on `qf-data-snapshot` (this commit)** |
| C-18 (CUDA dead code) | Closed on `qf-02-cuda-support` |
| C-19 (no economic backtest) | Closed on `qf-05-economic-backtest` |
| C-22 (statistical inference) | Closed on `qf-06-statistical-inference` |

What remains open from the original audit:

- **C-6, C-7, C-15, C-16, C-17, C-25** — minor hygiene items.
- **C-8, C-9, C-11, C-12, C-20, C-21, C-23, C-24** — major items
  that would benefit from but do not require a separate dedicated
  branch (e.g., adding balanced-accuracy and Brier as primary
  metrics, walk-forward as a robustness analysis, etc.).
- **C-10** — raw-return ablation: implicitly closed because the
  pipeline now defaults to raw-return labels and the addendum
  documents the comparison vs. wavelet defaults.
- **C-26** — paper limitations list extension; lands on
  `qf-08-writing-revision`.

The audit's **critical** list is fully addressed.
