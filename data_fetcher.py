"""
Fetch intraday stock data for NVDA and SK Hynix using yfinance.
Provides utilities for downloading, caching, and preparing data for prediction.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

TICKERS = {
    "NVDA": "NVDA",
    "HYNIX": "000660.KS",  # SK Hynix on Korea Exchange
}

CACHE_DIR = os.path.expanduser("~/.cache/stock_prediction")


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_intraday(ticker_key: str, period: str = "5d", interval: str = "1m") -> pd.DataFrame:
    """
    Fetch intraday data at 1-minute intervals.
    yfinance allows up to 7 days of 1m data, 60 days of 2m/5m data.
    """
    symbol = TICKERS[ticker_key]
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker_key} ({symbol}). Market may be closed.")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def fetch_historical(ticker_key: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetch longer historical data at 5-minute intervals for more training data.
    """
    symbol = TICKERS[ticker_key]
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker_key} ({symbol}).")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features."""
    df = df.copy()

    # Price-based features
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = pd.Series(df["Close"].values).apply(lambda x: x).pipe(
        lambda s: df["Close"].pct_change()
    )

    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"close_to_sma_{window}"] = df["Close"] / df[f"sma_{window}"] - 1

    # Exponential moving averages
    for span in [5, 10, 20]:
        df[f"ema_{span}"] = df["Close"].ewm(span=span).mean()

    # Volatility
    df["volatility_10"] = df["returns"].rolling(window=10).std()
    df["volatility_20"] = df["returns"].rolling(window=20).std()

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    bb_sma = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["bb_upper"] = bb_sma + 2 * bb_std
    df["bb_lower"] = bb_sma - 2 * bb_std
    df["bb_position"] = (df["Close"] - bb_lower) / (df["bb_upper"] - df["bb_lower"]).replace(0, 1e-10)

    # Volume features
    df["volume_sma_10"] = df["Volume"].rolling(window=10).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma_10"].replace(0, 1e-10)

    # Price momentum
    for lag in [1, 5, 10]:
        df[f"momentum_{lag}"] = df["Close"] / df["Close"].shift(lag) - 1

    # High-Low range
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    df.dropna(inplace=True)
    return df


def prepare_dataset(ticker_key: str, interval: str = "5m", lookback: int = 60,
                    horizon: int = 10) -> tuple:
    """
    Prepare dataset for training.

    Args:
        ticker_key: "NVDA" or "HYNIX"
        interval: data interval ("1m", "5m")
        lookback: number of past bars to use as input
        horizon: number of minutes ahead to predict

    Returns:
        (features_df, target_series, raw_df)
    """
    if interval == "1m":
        df = fetch_intraday(ticker_key, period="5d", interval="1m")
        steps_ahead = horizon  # 1m bars, so 10 steps = 10 minutes
    elif interval == "5m":
        df = fetch_historical(ticker_key, period="60d", interval="5m")
        steps_ahead = horizon // 5  # 5m bars, so 2 steps = 10 minutes
    else:
        raise ValueError(f"Unsupported interval: {interval}")

    df = add_technical_features(df)

    # Target: price change (%) after `steps_ahead` bars
    df["target"] = df["Close"].shift(-steps_ahead) / df["Close"] - 1
    df.dropna(inplace=True)

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in ["target"]]
    features = df[feature_cols]
    target = df["target"]

    return features, target, df


if __name__ == "__main__":
    for ticker in ["NVDA", "HYNIX"]:
        print(f"\n{'='*60}")
        print(f"Fetching data for {ticker}...")
        try:
            features, target, df = prepare_dataset(ticker, interval="5m")
            print(f"  Data shape: {features.shape}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Feature columns: {len(features.columns)}")
            print(f"  Target stats: mean={target.mean():.6f}, std={target.std():.6f}")
        except Exception as e:
            print(f"  Error: {e}")
