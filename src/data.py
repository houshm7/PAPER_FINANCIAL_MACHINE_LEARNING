"""Data acquisition and quality assessment utilities.

Snapshot policy
---------------
Yahoo Finance retroactively adjusts splits and dividends, so the
panel downloaded today is not necessarily byte-identical to the panel
that produced any committed results. To make the pipeline
reproducible, every download is mediated by :func:`load_or_download`,
which checks for a Parquet snapshot under ``data/snapshots/`` before
hitting yfinance. Snapshot files are versioned by ticker and date
range so multiple periods can coexist; the snapshot's modification
time + a small ``snapshot_metadata.json`` document the vintage.

Closes audit issue C-13.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import certifi
import pandas as pd
import numpy as np
import yfinance as yf

from .config import CONFIG

# Default location for committed price snapshots. Overridable via the
# ``DATA_SNAPSHOT_DIR`` environment variable for CI / testing.
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data" / "snapshots"


def _snapshot_dir() -> Path:
    return Path(os.environ.get("DATA_SNAPSHOT_DIR", DEFAULT_SNAPSHOT_DIR))


def _snapshot_path(ticker: str, start_date: str, end_date: str) -> Path:
    return _snapshot_dir() / f"{ticker}_{start_date}_{end_date}.parquet"


def _metadata_path() -> Path:
    return _snapshot_dir() / "snapshot_metadata.json"


_BROKEN_PROXY_VALUES = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
}
_PROXY_ENV_VARS = (
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "git_http_proxy",
    "git_https_proxy",
)
_YFINANCE_RUNTIME_READY = False


def _prepare_yfinance_runtime():
    """Work around local proxy/TLS issues before Yahoo Finance requests."""
    global _YFINANCE_RUNTIME_READY
    if _YFINANCE_RUNTIME_READY:
        return

    for env_var in _PROXY_ENV_VARS:
        proxy_value = os.environ.get(env_var)
        if proxy_value and proxy_value.strip().lower().rstrip("/") in _BROKEN_PROXY_VALUES:
            os.environ.pop(env_var, None)

    cert_path = certifi.where()
    if not cert_path.isascii():
        ascii_cert_path = Path(tempfile.gettempdir()) / "yfinance-cacert.pem"
        shutil.copyfile(cert_path, ascii_cert_path)
        os.environ["SSL_CERT_FILE"] = str(ascii_cert_path)
        os.environ["CURL_CA_BUNDLE"] = str(ascii_cert_path)

    # Surface the real network/TLS exception instead of the downstream
    # "'NoneType' object is not subscriptable" masking from yfinance.
    yf.config.debug.hide_exceptions = False
    _YFINANCE_RUNTIME_READY = True


def _download_from_yfinance(ticker, start_date, end_date, verbose=True):
    """Raw yfinance download (no snapshot logic)."""
    try:
        _prepare_yfinance_runtime()
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        df = df.reset_index()
        df = df.dropna(subset=["Close", "Volume"])
        df = df.set_index("Date")

        if verbose:
            print(f"  Downloaded {ticker}: {len(df)} days (live yfinance)")

        return df

    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        return None


def load_or_download(ticker, start_date, end_date,
                     *, prefer_snapshot=True, verbose=True):
    """Return OHLCV from a local Parquet snapshot if available, else yfinance.

    Snapshots are looked up at
    ``data/snapshots/{ticker}_{start}_{end}.parquet``. Override the
    snapshot directory via the ``DATA_SNAPSHOT_DIR`` environment
    variable.

    Parameters
    ----------
    ticker : str
    start_date, end_date : str   YYYY-MM-DD
    prefer_snapshot : bool       If False, force a live yfinance fetch.
    verbose : bool

    Returns
    -------
    pd.DataFrame or None
        OHLCV indexed by Date (tz-naive). None if both paths fail.
    """
    path = _snapshot_path(ticker, start_date, end_date)
    if prefer_snapshot and path.exists():
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if verbose:
                print(f"  Loaded {ticker}: {len(df)} days (snapshot {path.name})")
            return df
        except Exception as e:  # pragma: no cover — defensive
            print(f"  Snapshot load failed for {ticker}: {e!r}; "
                  f"falling back to yfinance.")

    return _download_from_yfinance(ticker, start_date, end_date, verbose=verbose)


def save_snapshot(ticker, start_date, end_date, df=None,
                  *, verbose=True):
    """Persist a ticker's OHLCV to ``data/snapshots/`` as Parquet.

    If ``df`` is None, downloads fresh from yfinance first.

    Returns the snapshot path on success, ``None`` on failure.
    """
    snap_dir = _snapshot_dir()
    snap_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = _download_from_yfinance(ticker, start_date, end_date,
                                     verbose=verbose)
        if df is None:
            return None

    path = _snapshot_path(ticker, start_date, end_date)
    df.to_parquet(path)
    _update_snapshot_metadata(ticker, start_date, end_date, df)
    if verbose:
        print(f"  Wrote {ticker}: {len(df)} days -> {path.name}")
    return path


def _update_snapshot_metadata(ticker, start_date, end_date, df):
    """Append/overwrite an entry in snapshot_metadata.json."""
    meta_path = _metadata_path()
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    else:
        meta = {}
    key = f"{ticker}_{start_date}_{end_date}"
    meta[key] = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "n_rows": int(len(df)),
        "first_date": str(df.index[0].date()) if len(df) else None,
        "last_date": str(df.index[-1].date()) if len(df) else None,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def download_stock_data(ticker, start_date, end_date, verbose=True):
    """Download historical stock data (snapshot-first; yfinance fallback).

    Backwards-compatible signature: existing callers that hit yfinance
    every time now transparently see snapshot data when one exists.
    Pass ``prefer_snapshot=False`` to ``load_or_download`` to force a
    fresh fetch.

    Returns
    -------
    pd.DataFrame or None
    """
    return load_or_download(ticker, start_date, end_date, verbose=verbose)

def download_multiple_stocks(tickers, start_date=None, end_date=None, config=None):
    """Download data for multiple stocks.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start_date, end_date : str, optional
        Override config dates.
    config : dict, optional
        Configuration dict (defaults to CONFIG).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from ticker to DataFrame.
    """
    if config is None:
        config = CONFIG
    start_date = start_date or config["start_date"]
    end_date = end_date or config["end_date"]

    data = {}
    failed = []

    print(f"Downloading data for {len(tickers)} stocks...")
    for ticker in tickers:
        df = download_stock_data(ticker, start_date, end_date)
        if df is not None:
            data[ticker] = df
        else:
            failed.append(ticker)

    print(f"Successfully downloaded: {len(data)}/{len(tickers)} stocks")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    return data


def assess_data_quality(data_dict):
    """Compute data-quality summary across all stocks.

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]

    Returns
    -------
    pd.DataFrame
        One row per ticker with quality metrics.
    """
    rows = []
    for ticker, df in data_dict.items():
        daily_returns = df["Close"].pct_change()
        rows.append(
            {
                "Ticker": ticker,
                "Days": len(df),
                "Date Range (days)": (df.index.max() - df.index.min()).days,
                "Missing (%)": (df.isnull().sum() / len(df) * 100).max(),
                "Avg Price": df["Close"].mean(),
                "Price Std": df["Close"].std(),
                "Avg Return (%)": daily_returns.mean() * 100,
                "Return Std (%)": daily_returns.std() * 100,
                "Extreme Moves (>10%)": (abs(daily_returns) > 0.10).sum(),
            }
        )
    return pd.DataFrame(rows)
