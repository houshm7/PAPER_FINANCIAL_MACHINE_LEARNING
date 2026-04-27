"""Data acquisition and quality assessment utilities."""

import os
import shutil
import tempfile
from pathlib import Path

import certifi
import pandas as pd
import numpy as np
import yfinance as yf

from .config import CONFIG


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


def download_stock_data(ticker, start_date, end_date, verbose=True):
    """Download historical stock data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start_date, end_date : str
        Date range in YYYY-MM-DD format.
    verbose : bool
        Print download status.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with OHLCV data indexed by Date, or None on failure.
    """
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
            print(f"  Downloaded {ticker}: {len(df)} days")

        return df

    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        return None

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
