"""Unit tests for the snapshot-first data layer (audit C-13).

Each test pins one piece of the contract:

1. ``save_snapshot`` writes a Parquet file at the expected path.
2. ``load_or_download`` returns the snapshot when it exists, without
   touching yfinance.
3. ``load_or_download`` falls back to yfinance when no snapshot
   exists (verified by monkey-patch).
4. The snapshot round-trip is value-preserving (Open/High/Low/Close/
   Volume identical to within Parquet's lossless rep).
5. ``snapshot_metadata.json`` is updated correctly.
6. The default snapshot path is overridable via the
   ``DATA_SNAPSHOT_DIR`` environment variable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.data as data_mod  # noqa: E402
from src.data import (  # noqa: E402
    _snapshot_path, load_or_download, save_snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_snapshot_dir(monkeypatch, tmp_path):
    """Redirect the snapshot directory to a clean tmp_path for each test."""
    monkeypatch.setenv("DATA_SNAPSHOT_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_ohlcv():
    idx = pd.bdate_range("2024-01-01", periods=20)
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, size=20)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "Open":  close * (1.0 + rng.normal(0, 0.001, 20)),
        "High":  close * (1.0 + rng.uniform(0, 0.01, 20)),
        "Low":   close * (1.0 - rng.uniform(0, 0.01, 20)),
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, 20).astype(float),
    }, index=idx)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# 1. save_snapshot writes the parquet at the expected path
# ---------------------------------------------------------------------------

def test_save_snapshot_writes_parquet(isolated_snapshot_dir, fake_ohlcv):
    path = save_snapshot("XYZ", "2024-01-01", "2024-01-31",
                         df=fake_ohlcv, verbose=False)
    assert path is not None
    assert path == _snapshot_path("XYZ", "2024-01-01", "2024-01-31")
    assert path.exists()
    assert path.suffix == ".parquet"


# ---------------------------------------------------------------------------
# 2. load_or_download returns snapshot when present, no yfinance call
# ---------------------------------------------------------------------------

def test_load_or_download_uses_snapshot(isolated_snapshot_dir, fake_ohlcv,
                                        monkeypatch):
    save_snapshot("XYZ", "2024-01-01", "2024-01-31",
                  df=fake_ohlcv, verbose=False)

    yf_calls = []
    def fake_dl(*a, **kw):
        yf_calls.append((a, kw))
        return None
    monkeypatch.setattr(data_mod, "_download_from_yfinance", fake_dl)

    df = load_or_download("XYZ", "2024-01-01", "2024-01-31", verbose=False)
    assert df is not None
    assert len(yf_calls) == 0, "yfinance must not be called when snapshot exists"
    assert len(df) == len(fake_ohlcv)
    pd.testing.assert_index_equal(df.index, fake_ohlcv.index)


# ---------------------------------------------------------------------------
# 3. Falls back to yfinance when no snapshot
# ---------------------------------------------------------------------------

def test_load_or_download_falls_back_to_yfinance(isolated_snapshot_dir,
                                                  fake_ohlcv, monkeypatch):
    yf_calls = []
    def fake_dl(ticker, start, end, verbose=True):
        yf_calls.append((ticker, start, end))
        return fake_ohlcv
    monkeypatch.setattr(data_mod, "_download_from_yfinance", fake_dl)

    df = load_or_download("ABC", "2024-01-01", "2024-01-31", verbose=False)
    assert df is not None
    assert len(yf_calls) == 1
    assert yf_calls[0] == ("ABC", "2024-01-01", "2024-01-31")


# ---------------------------------------------------------------------------
# 4. Round-trip is value-preserving
# ---------------------------------------------------------------------------

def test_snapshot_round_trip_preserves_values(isolated_snapshot_dir, fake_ohlcv):
    save_snapshot("XYZ", "2024-01-01", "2024-01-31",
                  df=fake_ohlcv, verbose=False)
    df_back = load_or_download("XYZ", "2024-01-01", "2024-01-31", verbose=False)
    assert df_back is not None
    for col in ("Open", "High", "Low", "Close", "Volume"):
        np.testing.assert_allclose(
            df_back[col].values, fake_ohlcv[col].values, rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# 5. snapshot_metadata.json is created/updated
# ---------------------------------------------------------------------------

def test_metadata_recorded(isolated_snapshot_dir, fake_ohlcv):
    save_snapshot("XYZ", "2024-01-01", "2024-01-31",
                  df=fake_ohlcv, verbose=False)
    meta_path = isolated_snapshot_dir / "snapshot_metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    key = "XYZ_2024-01-01_2024-01-31"
    assert key in meta
    entry = meta[key]
    assert entry["ticker"] == "XYZ"
    assert entry["n_rows"] == len(fake_ohlcv)
    assert entry["first_date"] == "2024-01-01"
    # last_date == last business day in the 20-day window
    assert entry["last_date"] == "2024-01-26"
    assert "saved_utc" in entry


# ---------------------------------------------------------------------------
# 6. DATA_SNAPSHOT_DIR override works
# ---------------------------------------------------------------------------

def test_env_var_override(tmp_path, monkeypatch, fake_ohlcv):
    custom = tmp_path / "elsewhere"
    monkeypatch.setenv("DATA_SNAPSHOT_DIR", str(custom))
    path = save_snapshot("XYZ", "2024-01-01", "2024-01-31",
                         df=fake_ohlcv, verbose=False)
    assert path is not None
    assert custom in path.parents


# ---------------------------------------------------------------------------
# 7. download_stock_data is a thin wrapper over load_or_download
# ---------------------------------------------------------------------------

def test_download_stock_data_uses_snapshot(isolated_snapshot_dir,
                                           fake_ohlcv, monkeypatch):
    save_snapshot("XYZ", "2024-01-01", "2024-01-31",
                  df=fake_ohlcv, verbose=False)
    yf_calls = []
    def fake_dl(*a, **kw):
        yf_calls.append(a)
        return None
    monkeypatch.setattr(data_mod, "_download_from_yfinance", fake_dl)
    df = data_mod.download_stock_data("XYZ", "2024-01-01", "2024-01-31",
                                      verbose=False)
    assert df is not None
    assert yf_calls == []


# ---------------------------------------------------------------------------
# 8. prefer_snapshot=False forces a live fetch
# ---------------------------------------------------------------------------

def test_prefer_snapshot_false_skips_cache(isolated_snapshot_dir,
                                            fake_ohlcv, monkeypatch):
    save_snapshot("XYZ", "2024-01-01", "2024-01-31",
                  df=fake_ohlcv, verbose=False)
    yf_calls = []
    def fake_dl(*a, **kw):
        yf_calls.append(a)
        return fake_ohlcv
    monkeypatch.setattr(data_mod, "_download_from_yfinance", fake_dl)
    df = load_or_download("XYZ", "2024-01-01", "2024-01-31",
                          prefer_snapshot=False, verbose=False)
    assert df is not None
    assert len(yf_calls) == 1
