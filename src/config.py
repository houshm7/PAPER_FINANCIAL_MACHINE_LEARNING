"""Project configuration: stock universe, hyperparameters, and constants."""

STOCK_UNIVERSE = {
    "Technology": {
        "stocks": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        "description": "Leading technology companies",
    },
    "Automotive": {
        "stocks": ["TSLA", "F", "GM", "TM", "HMC"],
        "description": "Major automotive manufacturers",
    },
    "Consumer": {
        "stocks": ["NKE", "SBUX", "MCD", "DIS", "NFLX"],
        "description": "Consumer goods and entertainment",
    },
    "Financial": {
        "stocks": ["JPM", "BAC", "GS", "MS", "C"],
        "description": "Major financial institutions",
    },
    "Healthcare": {
        "stocks": ["JNJ", "UNH", "PFE", "ABBV", "TMO"],
        "description": "Healthcare and pharmaceutical companies",
    },
}

CONFIG = {
    # Date range
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    # Preprocessing — Smoothing
    "alpha": 0.095,          # Exponential smoothing factor (Basak et al., 2019)
    "wavelet": "db4",        # Wavelet family for DWT denoising
    "wavelet_level": None,   # Decomposition level (None = auto)
    "savgol_window": 21,     # Savitzky-Golay window length (must be odd)
    "savgol_polyorder": 3,   # Savitzky-Golay polynomial order
    # Technical indicators — Original (Basak et al., 2019)
    "rsi_period": 14,
    "so_period": 14,
    "wr_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "proc_period": 10,
    # Technical indicators — Extended
    "bb_period": 20,      # Bollinger Bands lookback
    "bb_std": 2,          # Bollinger Bands standard deviations
    "atr_period": 14,     # Average True Range
    "adx_period": 14,     # Average Directional Index
    "mfi_period": 14,     # Money Flow Index
    "cci_period": 20,     # Commodity Channel Index
    "hvol_period": 20,    # Historical Volatility
    # Trading windows (in days)
    "windows": [1, 2, 5, 10, 15],
    # Model parameters
    "n_estimators": 100,
    "test_size": 0.2,
    "random_state": 42,
    "n_jobs": -1,
}

# Original 6 features from Basak et al. (2019)
ORIGINAL_FEATURE_COLS = ["RSI", "SO", "WR", "MACD", "PROC", "OBV"]

# Extended features: original + volatility + trend + volume + cyclical
EXTENDED_FEATURE_COLS = [
    # Momentum (original)
    "RSI", "SO", "WR", "MACD", "PROC",
    # Volume (original + extended)
    "OBV", "MFI", "AD_LINE",
    # Volatility (new)
    "BB_WIDTH", "BB_PCT", "ATR", "HIST_VOL",
    # Trend strength (new)
    "ADX",
    # Cyclical (new)
    "CCI",
]

# Change (Δ) features: 1-day change for each indicator — captures momentum dynamics
CHANGE_FEATURE_COLS = [f"{col}_CHG" for col in EXTENDED_FEATURE_COLS]

# Default feature set — use extended
FEATURE_COLS = EXTENDED_FEATURE_COLS

TREE_MODEL_NAMES = ["Random Forest", "XGBoost", "Gradient Boosting", "LightGBM", "CatBoost"]

DL_MODEL_NAMES = ["MLP", "LSTM"]

MODEL_NAMES = TREE_MODEL_NAMES

ENSEMBLE_NAMES = ["Stacking"]

BASELINE_NAMES = ["Dummy (Most Frequent)", "Dummy (Stratified)", "Logistic Regression"]

ALL_MODEL_NAMES = BASELINE_NAMES + MODEL_NAMES + ENSEMBLE_NAMES

MODEL_COLORS = {
    "Dummy (Most Frequent)": "#999999",
    "Dummy (Stratified)": "#bdbdbd",
    "Logistic Regression": "#636363",
    "Random Forest": "#e41a1c",
    "XGBoost": "#377eb8",
    "Gradient Boosting": "#4daf4a",
    "LightGBM": "#984ea3",
    "CatBoost": "#ff7f00",
    "MLP": "#a65628",
    "LSTM": "#f781bf",
    "Stacking": "#e31a1c",
}

MODEL_MARKERS = {
    "Dummy (Most Frequent)": "x",
    "Dummy (Stratified)": "+",
    "Logistic Regression": "p",
    "Random Forest": "o",
    "XGBoost": "s",
    "Gradient Boosting": "^",
    "LightGBM": "D",
    "CatBoost": "v",
    "MLP": "P",
    "LSTM": "*",
    "Stacking": "H",
}


def get_all_tickers():
    """Return a flat list of all tickers across sectors."""
    tickers = []
    for sector_info in STOCK_UNIVERSE.values():
        tickers.extend(sector_info["stocks"])
    return tickers


def get_ticker_to_sector_map():
    """Return a dict mapping each ticker to its sector name."""
    mapping = {}
    for sector, info in STOCK_UNIVERSE.items():
        for ticker in info["stocks"]:
            mapping[ticker] = sector
    return mapping
