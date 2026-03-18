#!/usr/bin/env python3
"""
Stock Price Prediction — 10 Minutes Ahead
Predicts NVDA and SK Hynix (000660.KS) prices using LSTM/Transformer models.

Usage:
    uv run predict.py                    # Train & predict both stocks
    uv run predict.py --ticker NVDA      # Train & predict NVDA only
    uv run predict.py --ticker HYNIX     # Train & predict SK Hynix only
    uv run predict.py --model transformer # Use Transformer instead of LSTM
    uv run predict.py --live             # Continuous live prediction mode
"""

import argparse
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from data_fetcher import fetch_intraday, fetch_historical, add_technical_features, TICKERS
from model import StockLSTM, StockTransformer, create_sequences

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOOKBACK = 60        # Number of past bars as input
HORIZON_MIN = 10     # Predict 10 minutes ahead
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
TRAIN_SPLIT = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints"
RESULTS_DIR = "results"


def prepare_data(ticker_key: str, interval: str = "5m"):
    """Fetch data, add features, create sequences."""
    print(f"  Fetching {ticker_key} data ({interval} interval)...")

    if interval == "1m":
        df = fetch_intraday(ticker_key, period="5d", interval="1m")
        steps_ahead = HORIZON_MIN
    else:
        df = fetch_historical(ticker_key, period="60d", interval="5m")
        steps_ahead = max(1, HORIZON_MIN // 5)

    df = add_technical_features(df)

    # Target: future price change (%)
    df["target"] = df["Close"].shift(-steps_ahead) / df["Close"] - 1
    df.dropna(inplace=True)

    feature_cols = [c for c in df.columns if c != "target"]
    features = df[feature_cols].values
    targets = df["target"].values
    close_prices = df["Close"].values
    timestamps = df.index

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create sequences
    X, y = create_sequences(features_scaled, targets, lookback=LOOKBACK)

    # Align close prices and timestamps with sequences
    close_seq = close_prices[LOOKBACK:]
    ts_seq = timestamps[LOOKBACK:]

    return X, y, close_seq, ts_seq, scaler, feature_cols, df


def train_model(X_train, y_train, X_val, y_val, input_size, model_type="lstm"):
    """Train the prediction model."""
    if model_type == "lstm":
        model = StockLSTM(input_size=input_size, hidden_size=128, num_layers=3, dropout=0.2)
    else:
        model = StockTransformer(input_size=input_size, d_model=128, nhead=4, num_layers=3, dropout=0.2)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=0.5)

    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate_model(model, X_val, y_val, close_val, ts_val, ticker_key):
    """Evaluate model and generate predictions."""
    model.eval()
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)

    with torch.no_grad():
        pred_pct = model(X_val_t).cpu().numpy()

    actual_pct = y_val

    # Convert percentage predictions to price predictions
    pred_prices = close_val * (1 + pred_pct)
    actual_prices = close_val * (1 + actual_pct)

    # Metrics
    mae_pct = np.mean(np.abs(pred_pct - actual_pct)) * 100
    rmse_pct = np.sqrt(np.mean((pred_pct - actual_pct) ** 2)) * 100
    mae_price = np.mean(np.abs(pred_prices - actual_prices))

    # Directional accuracy
    pred_direction = (pred_pct > 0).astype(int)
    actual_direction = (actual_pct > 0).astype(int)
    dir_accuracy = np.mean(pred_direction == actual_direction) * 100

    print(f"\n  === {ticker_key} Evaluation ===")
    print(f"  MAE (% change):      {mae_pct:.4f}%")
    print(f"  RMSE (% change):     {rmse_pct:.4f}%")
    print(f"  MAE (price):         ${mae_price:.2f}")
    print(f"  Direction accuracy:  {dir_accuracy:.1f}%")
    print(f"  Latest close:        ${close_val[-1]:.2f}")
    print(f"  Predicted change:    {pred_pct[-1]*100:+.4f}%")
    print(f"  Predicted price:     ${pred_prices[-1]:.2f} (in 10 min)")

    return {
        "ticker": ticker_key,
        "mae_pct": mae_pct,
        "rmse_pct": rmse_pct,
        "mae_price": mae_price,
        "direction_accuracy": dir_accuracy,
        "last_close": close_val[-1],
        "predicted_change_pct": pred_pct[-1] * 100,
        "predicted_price": pred_prices[-1],
        "predictions": pred_pct,
        "actuals": actual_pct,
        "close_prices": close_val,
        "timestamps": ts_val,
    }


def plot_results(results: dict, ticker_key: str):
    """Generate prediction plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f"{ticker_key} Stock Price Prediction (10-min ahead)", fontsize=14, fontweight="bold")

    pred = results["predictions"]
    actual = results["actuals"]
    close = results["close_prices"]
    n = len(pred)
    x = np.arange(n)

    # Plot 1: Predicted vs actual price changes
    axes[0].plot(x[-200:], actual[-200:] * 100, label="Actual", alpha=0.7, linewidth=1)
    axes[0].plot(x[-200:], pred[-200:] * 100, label="Predicted", alpha=0.7, linewidth=1)
    axes[0].set_title("Predicted vs Actual Price Change (%)")
    axes[0].set_ylabel("Price Change (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Predicted vs actual prices
    pred_prices = close[-200:] * (1 + pred[-200:])
    actual_prices = close[-200:] * (1 + actual[-200:])
    axes[1].plot(x[-200:], actual_prices, label="Actual Future Price", alpha=0.7, linewidth=1)
    axes[1].plot(x[-200:], pred_prices, label="Predicted Future Price", alpha=0.7, linewidth=1)
    axes[1].plot(x[-200:], close[-200:], label="Current Price", alpha=0.5, linewidth=1, linestyle="--")
    axes[1].set_title("Price Prediction")
    axes[1].set_ylabel("Price ($)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Prediction error
    error = (pred[-200:] - actual[-200:]) * 100
    axes[2].bar(x[-200:], error, alpha=0.6, width=1.0, color=np.where(error > 0, "red", "blue"))
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].set_title("Prediction Error (%)")
    axes[2].set_ylabel("Error (%)")
    axes[2].set_xlabel("Time Step")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{ticker_key}_prediction.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved to {path}")


def run_prediction(ticker_key: str, model_type: str = "lstm", interval: str = "5m"):
    """Full pipeline: fetch data, train, evaluate, predict."""
    print(f"\n{'='*60}")
    print(f"  {ticker_key} — 10-Minute Ahead Prediction")
    print(f"  Model: {model_type.upper()} | Interval: {interval} | Device: {DEVICE}")
    print(f"{'='*60}")

    # Prepare data
    X, y, close, timestamps, scaler, feature_cols, df = prepare_data(ticker_key, interval)
    print(f"  Total sequences: {len(X)}")
    print(f"  Features: {len(feature_cols)}")

    # Train/val split
    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    close_train, close_val = close[:split], close[split:]
    ts_train, ts_val = timestamps[:split], timestamps[split:]

    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"\n  Training {model_type.upper()} model...")

    # Train
    model, val_loss = train_model(X_train, y_train, X_val, y_val,
                                   input_size=X.shape[2], model_type=model_type)

    # Save checkpoint
    os.makedirs(SAVE_DIR, exist_ok=True)
    ckpt_path = os.path.join(SAVE_DIR, f"{ticker_key}_{model_type}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": feature_cols,
        "model_type": model_type,
        "input_size": X.shape[2],
    }, ckpt_path)
    print(f"  Model saved to {ckpt_path}")

    # Evaluate
    results = evaluate_model(model, X_val, y_val, close_val, ts_val, ticker_key)

    # Plot
    plot_results(results, ticker_key)

    return results


def live_prediction(ticker_key: str, model_type: str = "lstm", refresh_sec: int = 60):
    """
    Continuous live prediction mode.
    Fetches latest data every `refresh_sec` seconds and makes predictions.
    """
    ckpt_path = os.path.join(SAVE_DIR, f"{ticker_key}_{model_type}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint found at {ckpt_path}. Training first...")
        run_prediction(ticker_key, model_type)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    input_size = checkpoint["input_size"]
    feature_cols = checkpoint["feature_cols"]

    if model_type == "lstm":
        model = StockLSTM(input_size=input_size)
    else:
        model = StockTransformer(input_size=input_size)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]

    print(f"\n  Live prediction mode for {ticker_key}")
    print(f"  Refreshing every {refresh_sec}s. Press Ctrl+C to stop.\n")

    predictions_log = []

    try:
        while True:
            try:
                df = fetch_intraday(ticker_key, period="1d", interval="1m")
                df = add_technical_features(df)

                feat_cols_available = [c for c in feature_cols if c in df.columns]
                if len(feat_cols_available) < len(feature_cols):
                    print(f"  Warning: missing {len(feature_cols) - len(feat_cols_available)} features")

                features = df[feat_cols_available].values
                features_scaled = scaler.transform(features)

                if len(features_scaled) >= LOOKBACK:
                    seq = features_scaled[-LOOKBACK:]
                    seq_t = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        pred_pct = model(seq_t).item()

                    current_price = df["Close"].iloc[-1]
                    predicted_price = current_price * (1 + pred_pct)
                    direction = "UP" if pred_pct > 0 else "DOWN"

                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{now}] {ticker_key}: ${current_price:.2f} -> "
                          f"${predicted_price:.2f} ({pred_pct*100:+.4f}%) {direction}")

                    predictions_log.append({
                        "time": now,
                        "current": current_price,
                        "predicted": predicted_price,
                        "change_pct": pred_pct * 100,
                    })
                else:
                    print(f"  Not enough data yet ({len(features_scaled)} < {LOOKBACK} bars)")

            except Exception as e:
                print(f"  Error: {e}")

            time.sleep(refresh_sec)

    except KeyboardInterrupt:
        print("\n  Stopped.")
        if predictions_log:
            log_df = pd.DataFrame(predictions_log)
            log_path = os.path.join(RESULTS_DIR, f"{ticker_key}_live_log.csv")
            os.makedirs(RESULTS_DIR, exist_ok=True)
            log_df.to_csv(log_path, index=False)
            print(f"  Live predictions saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Stock Price Prediction — 10 Minutes Ahead")
    parser.add_argument("--ticker", type=str, default="all",
                        choices=["NVDA", "HYNIX", "all"],
                        help="Which stock to predict (default: all)")
    parser.add_argument("--model", type=str, default="lstm",
                        choices=["lstm", "transformer"],
                        help="Model architecture (default: lstm)")
    parser.add_argument("--interval", type=str, default="5m",
                        choices=["1m", "5m"],
                        help="Data interval (default: 5m for more training data)")
    parser.add_argument("--live", action="store_true",
                        help="Enable live prediction mode")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    args = parser.parse_args()

    global EPOCHS
    EPOCHS = args.epochs

    tickers = list(TICKERS.keys()) if args.ticker == "all" else [args.ticker]

    if args.live:
        if len(tickers) > 1:
            print("Live mode supports one ticker at a time. Use --ticker NVDA or --ticker HYNIX")
            sys.exit(1)
        live_prediction(tickers[0], model_type=args.model)
    else:
        all_results = {}
        for ticker in tickers:
            results = run_prediction(ticker, model_type=args.model, interval=args.interval)
            all_results[ticker] = results

        # Print summary
        print(f"\n{'='*60}")
        print("  PREDICTION SUMMARY")
        print(f"{'='*60}")
        for ticker, r in all_results.items():
            print(f"\n  {ticker}:")
            print(f"    Current Price:       ${r['last_close']:.2f}")
            print(f"    Predicted Change:    {r['predicted_change_pct']:+.4f}%")
            print(f"    Predicted Price:     ${r['predicted_price']:.2f} (in 10 min)")
            print(f"    Direction Accuracy:  {r['direction_accuracy']:.1f}%")
            print(f"    MAE:                 {r['mae_pct']:.4f}%")


if __name__ == "__main__":
    main()
