"""
Stock price prediction data preparation and evaluation utilities.
Downloads intraday data for NVDA and SK Hynix, computes technical features,
and provides fixed evaluation metrics.

Usage:
    python prepare.py                  # download data for both tickers
    python prepare.py --ticker NVDA    # download NVDA only

Data is stored in ~/.cache/stock_prediction/.
This file is READ-ONLY — do not modify.
"""

import os
import sys
import time
import math
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

LOOKBACK = 60            # number of past bars as model input
HORIZON_MINUTES = 10     # predict 10 minutes ahead
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
TRAIN_SPLIT = 0.8        # 80% train, 20% val
EVAL_BATCHES = 50        # number of batches for validation evaluation

TICKERS = {
    "NVDA": "NVDA",              # NVIDIA on NASDAQ
    "HYNIX": "000660.KS",        # SK Hynix on Korea Exchange
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "stock_prediction")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_stock_data(ticker_key: str, interval: str = "5m", period: str = "60d"):
    """Download stock data and save to cache."""
    os.makedirs(DATA_DIR, exist_ok=True)
    symbol = TICKERS[ticker_key]
    cache_file = os.path.join(DATA_DIR, f"{ticker_key}_{interval}.pkl")

    print(f"  Downloading {ticker_key} ({symbol}) — {interval} interval, {period} period...")
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df.empty:
                raise ValueError(f"No data returned for {ticker_key} ({symbol})")
            # Flatten multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            df.to_pickle(cache_file)
            print(f"  Downloaded {len(df)} bars for {ticker_key}, saved to {cache_file}")
            return df
        except Exception as e:
            print(f"  Attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    print(f"  FAILED to download {ticker_key}")
    return None


def load_cached_data(ticker_key: str, interval: str = "5m") -> pd.DataFrame:
    """Load cached stock data."""
    cache_file = os.path.join(DATA_DIR, f"{ticker_key}_{interval}.pkl")
    if not os.path.exists(cache_file):
        return None
    return pd.read_pickle(cache_file)

# ---------------------------------------------------------------------------
# Feature engineering (fixed — do not modify)
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators. Returns DataFrame with feature columns."""
    df = df.copy()

    # Returns
    df["returns"] = df["Close"].pct_change()

    # Moving averages
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["Close"].rolling(window=w).mean()
        df[f"close_to_sma_{w}"] = df["Close"] / df[f"sma_{w}"] - 1

    # Exponential moving averages
    for span in [5, 10, 20]:
        df[f"ema_{span}"] = df["Close"].ewm(span=span).mean()

    # Volatility
    df["volatility_10"] = df["returns"].rolling(window=10).std()
    df["volatility_20"] = df["returns"].rolling(window=20).std()

    # RSI
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
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, 1e-10)
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / bb_range

    # Volume features
    df["volume_sma_10"] = df["Volume"].rolling(window=10).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma_10"].replace(0, 1e-10)

    # Momentum
    for lag in [1, 5, 10]:
        df[f"momentum_{lag}"] = df["Close"] / df["Close"].shift(lag) - 1

    # High-Low range
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    df.dropna(inplace=True)
    return df

# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

FEATURE_COLS = None  # Set after first prepare_dataset call


def prepare_dataset(ticker_key: str, interval: str = "5m"):
    """
    Prepare train/val datasets for a ticker.

    Returns:
        dict with keys: X_train, y_train, X_val, y_val, close_train, close_val,
                        feature_cols, scaler_mean, scaler_std, n_features
    """
    global FEATURE_COLS

    df = load_cached_data(ticker_key, interval)
    if df is None:
        df = download_stock_data(ticker_key, interval)
    if df is None or len(df) < LOOKBACK + 50:
        raise ValueError(f"Not enough data for {ticker_key}")

    df = add_features(df)

    # Compute target: future % change
    if interval == "1m":
        steps_ahead = HORIZON_MINUTES
    elif interval == "5m":
        steps_ahead = max(1, HORIZON_MINUTES // 5)
    else:
        steps_ahead = max(1, HORIZON_MINUTES // 5)

    df["target"] = df["Close"].shift(-steps_ahead) / df["Close"] - 1
    df.dropna(inplace=True)

    # Feature columns (everything except target)
    feature_cols = [c for c in df.columns if c != "target"]
    FEATURE_COLS = feature_cols

    features = df[feature_cols].values.astype(np.float32)
    targets = df["target"].values.astype(np.float32)
    close_prices = df["Close"].values.astype(np.float32)

    # Normalize features (z-score)
    scaler_mean = features.mean(axis=0)
    scaler_std = features.std(axis=0) + 1e-8
    features_scaled = (features - scaler_mean) / scaler_std

    # Create sliding window sequences
    X, y, close = [], [], []
    for i in range(LOOKBACK, len(features_scaled)):
        X.append(features_scaled[i - LOOKBACK:i])
        y.append(targets[i])
        close.append(close_prices[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    close = np.array(close, dtype=np.float32)

    # Train/val split
    split = int(len(X) * TRAIN_SPLIT)
    return {
        "X_train": X[:split], "y_train": y[:split],
        "X_val": X[split:], "y_val": y[split:],
        "close_train": close[:split], "close_val": close[split:],
        "feature_cols": feature_cols,
        "scaler_mean": scaler_mean, "scaler_std": scaler_std,
        "n_features": X.shape[2],
    }


def make_dataloader(X, y, batch_size, shuffle=True):
    """
    Infinite dataloader that yields (x_batch, y_batch) tensors on CUDA.

    Args:
        X: numpy array (N, lookback, n_features)
        y: numpy array (N,)
        batch_size: batch size
        shuffle: whether to shuffle each epoch
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    N = len(X_t)
    epoch = 1

    while True:
        if shuffle:
            perm = torch.randperm(N, device=device)
            X_t = X_t[perm]
            y_t = y_t[perm]
        for i in range(0, N - batch_size + 1, batch_size):
            yield X_t[i:i + batch_size], y_t[i:i + batch_size], epoch
        epoch += 1


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, X_val, y_val, close_val, batch_size):
    """
    Fixed evaluation: computes val_mae (%), val_rmse (%), and direction_accuracy (%).
    The primary metric is val_mae — lower is better.

    Args:
        model: PyTorch model that takes (batch, lookback, features) -> (batch,)
        X_val, y_val, close_val: numpy arrays
        batch_size: evaluation batch size

    Returns:
        dict with val_mae, val_rmse, direction_accuracy
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    all_preds = []
    all_targets = []

    N = len(X_val)
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X_val[i:i + batch_size]).to(device)
        yb = y_val[i:i + batch_size]
        pred = model(xb).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(yb)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # MAE and RMSE in percentage points
    mae = np.mean(np.abs(preds - targets)) * 100
    rmse = np.sqrt(np.mean((preds - targets) ** 2)) * 100

    # Directional accuracy
    pred_dir = (preds > 0).astype(int)
    actual_dir = (targets > 0).astype(int)
    direction_acc = np.mean(pred_dir == actual_dir) * 100

    return {
        "val_mae": mae,
        "val_rmse": rmse,
        "direction_accuracy": direction_acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare stock data for autoresearch")
    parser.add_argument("--ticker", type=str, default="all", choices=["NVDA", "HYNIX", "all"])
    parser.add_argument("--interval", type=str, default="5m", choices=["1m", "5m"])
    args = parser.parse_args()

    tickers = list(TICKERS.keys()) if args.ticker == "all" else [args.ticker]

    print(f"Cache directory: {CACHE_DIR}")
    print()

    for ticker in tickers:
        df = download_stock_data(ticker, interval=args.interval)
        if df is not None:
            df = add_features(df)
            print(f"  {ticker}: {len(df)} bars with features, {len(df.columns)} columns")
        print()

    print("Done! Ready to train.")
