"""Technical indicator calculations.

Original (Basak et al., 2019): RSI, SO, WR, MACD, PROC, OBV
Extended: Bollinger Bands, ATR, ADX, MFI, A/D Line, CCI, Historical Volatility
"""

import numpy as np
import pandas as pd

from .config import CONFIG


def calculate_rsi(prices, period=14):
    """Relative Strength Index (Wilder, 1978).

    Parameters
    ----------
    prices : pd.Series  — closing prices.
    period : int         — look-back window.

    Returns
    -------
    pd.Series
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic_oscillator(high, low, close, period=14):
    """Stochastic Oscillator %K (Lane, 1950s).

    Parameters
    ----------
    high, low, close : pd.Series
    period : int

    Returns
    -------
    pd.Series
    """
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    return 100 * (close - lowest_low) / (highest_high - lowest_low)


def calculate_williams_r(high, low, close, period=14):
    """Williams %R (Williams, 1973).

    Parameters
    ----------
    high, low, close : pd.Series
    period : int

    Returns
    -------
    pd.Series
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD histogram (Appel, late 1970s).

    Parameters
    ----------
    prices : pd.Series — closing prices.
    fast, slow, signal : int — EMA periods.

    Returns
    -------
    pd.Series — MACD line minus signal line.
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def calculate_proc(prices, period=10):
    """Price Rate of Change.

    Parameters
    ----------
    prices : pd.Series
    period : int

    Returns
    -------
    pd.Series — percentage change over *period* days.
    """
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


def calculate_obv(close, volume):
    """On Balance Volume (Granville, 1960s).

    Parameters
    ----------
    close : pd.Series
    volume : pd.Series

    Returns
    -------
    pd.Series
    """
    price_change = close.diff()
    direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    obv = (direction * volume).cumsum()
    return pd.Series(obv, index=close.index)


# ---------------------------------------------------------------------------
# Extended indicators (volatility, trend strength, volume, cyclical)
# ---------------------------------------------------------------------------

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Bollinger Bands (Bollinger, 1980s).

    Returns
    -------
    bb_width : pd.Series — band width (volatility measure).
    bb_pct : pd.Series — %B (price position within bands, 0-1).
    """
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std

    bb_width = (upper - lower) / sma
    bb_pct = (prices - lower) / (upper - lower)
    return bb_width, bb_pct


def calculate_atr(high, low, close, period=14):
    """Average True Range (Wilder, 1978) — measures volatility magnitude.

    Parameters
    ----------
    high, low, close : pd.Series
    period : int

    Returns
    -------
    pd.Series
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_adx(high, low, close, period=14):
    """Average Directional Index (Wilder, 1978) — measures trend strength.

    ADX > 25: strong trend, ADX < 20: weak/no trend.

    Returns
    -------
    pd.Series
    """
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = calculate_atr(high, low, close, period)

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx


def calculate_mfi(high, low, close, volume, period=14):
    """Money Flow Index (Quong & Soudack, 1989) — volume-weighted RSI.

    MFI > 80: overbought, MFI < 20: oversold.

    Returns
    -------
    pd.Series
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    delta = typical_price.diff()

    positive_flow = money_flow.where(delta > 0, 0.0).rolling(window=period).sum()
    negative_flow = money_flow.where(delta <= 0, 0.0).rolling(window=period).sum()

    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    return mfi


def calculate_ad_line(high, low, close, volume):
    """Accumulation/Distribution Line (Williams, 1972) — buying/selling pressure.

    More nuanced than OBV: weights volume by where close falls within the
    high-low range rather than using a binary up/down signal.

    Returns
    -------
    pd.Series
    """
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    ad = (clv * volume).cumsum()
    return pd.Series(ad, index=close.index)


def calculate_cci(high, low, close, period=20):
    """Commodity Channel Index (Lambert, 1980) — cyclical price deviation.

    CCI > +100: overbought / strong uptrend.
    CCI < -100: oversold / strong downtrend.

    Returns
    -------
    pd.Series
    """
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (typical_price - sma) / (0.015 * mad)


def calculate_historical_volatility(prices, period=20):
    """Historical Volatility — annualized standard deviation of log returns.

    Returns
    -------
    pd.Series
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=period).std() * np.sqrt(252)


def calculate_all_indicators(df, config=None, extended=True):
    """Calculate technical indicators on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Open, High, Low, Close, Volume columns.
    config : dict, optional
    extended : bool
        If True (default), compute all 15 indicators (original + extended).
        If False, compute only the 6 original Basak et al. indicators.

    Returns
    -------
    pd.DataFrame — copy of *df* with indicator columns added.
    """
    if config is None:
        config = CONFIG

    result = df.copy()

    # --- Original indicators (Basak et al., 2019) ---
    result["RSI"] = calculate_rsi(result["Close"], period=config["rsi_period"])
    result["SO"] = calculate_stochastic_oscillator(
        result["High"], result["Low"], result["Close"], period=config["so_period"]
    )
    result["WR"] = calculate_williams_r(
        result["High"], result["Low"], result["Close"], period=config["wr_period"]
    )
    result["MACD"] = calculate_macd(
        result["Close"],
        fast=config["macd_fast"],
        slow=config["macd_slow"],
        signal=config["macd_signal"],
    )
    result["PROC"] = calculate_proc(result["Close"], period=config["proc_period"])
    result["OBV"] = calculate_obv(result["Close"], result["Volume"])

    if not extended:
        return result

    # --- Extended indicators ---
    # Volatility
    bb_width, bb_pct = calculate_bollinger_bands(
        result["Close"], period=config["bb_period"], num_std=config["bb_std"]
    )
    result["BB_WIDTH"] = bb_width
    result["BB_PCT"] = bb_pct
    result["ATR"] = calculate_atr(
        result["High"], result["Low"], result["Close"], period=config["atr_period"]
    )
    result["HIST_VOL"] = calculate_historical_volatility(
        result["Close"], period=config["hvol_period"]
    )

    # Trend strength
    result["ADX"] = calculate_adx(
        result["High"], result["Low"], result["Close"], period=config["adx_period"]
    )

    # Volume-weighted
    result["MFI"] = calculate_mfi(
        result["High"], result["Low"], result["Close"], result["Volume"],
        period=config["mfi_period"]
    )
    result["AD_LINE"] = calculate_ad_line(
        result["High"], result["Low"], result["Close"], result["Volume"]
    )

    # Cyclical
    result["CCI"] = calculate_cci(
        result["High"], result["Low"], result["Close"], period=config["cci_period"]
    )

    return result
