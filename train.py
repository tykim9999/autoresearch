"""
Stock price prediction training script. Single-GPU, single-file.
Predicts NVDA and SK Hynix prices 10 minutes ahead.

This is the file the agent modifies. Everything is fair game:
model architecture, optimizer, hyperparameters, training loop, batch size, etc.

Usage: uv run train.py
"""

import os
import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    LOOKBACK, TIME_BUDGET, TICKERS,
    prepare_dataset, make_dataloader, evaluate,
)

# ---------------------------------------------------------------------------
# Model Architecture (edit freely)
# ---------------------------------------------------------------------------

class StockLSTM(nn.Module):
    """Multi-layer LSTM with attention for stock price prediction."""

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Attention over time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, lookback, features)
        lstm_out, _ = self.lstm(x)  # (batch, lookback, hidden)
        # Attention
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, lookback, 1)
        context = (lstm_out * attn_w).sum(dim=1)  # (batch, hidden)
        return self.head(context).squeeze(-1)  # (batch,)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — no CLI flags needed)
# ---------------------------------------------------------------------------

# Model
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.2

# Optimization
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.05
EPOCHS_MAX = 200        # max epochs (usually stops by time budget)

# Data
INTERVAL = "5m"         # "1m" or "5m"

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ticker(ticker_key: str):
    """Train model for one ticker. Returns evaluation metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  {ticker_key} — 10-Minute Ahead Stock Price Prediction")
    print(f"  Device: {device} | Interval: {INTERVAL}")
    print(f"{'='*60}")

    # Prepare data
    data = prepare_dataset(ticker_key, interval=INTERVAL)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    close_val = data["close_val"]
    n_features = data["n_features"]

    print(f"  Features: {n_features}")
    print(f"  Train: {len(X_train)} sequences | Val: {len(X_val)} sequences")

    # Build model
    model = StockLSTM(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.HuberLoss(delta=0.5)

    # Dataloader
    loader = make_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)

    # LR schedule
    total_steps = EPOCHS_MAX * steps_per_epoch
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=WARMUP_RATIO, anneal_strategy="cos",
    )

    # Training
    print(f"  Time budget: {TIME_BUDGET}s")
    print(f"  Training...")

    t_start = time.time()
    total_training_time = 0.0
    step = 0
    best_val_mae = float("inf")
    best_state = None
    smooth_loss = 0.0

    while True:
        model.train()
        t0 = time.time()

        xb, yb, epoch = next(loader)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step < total_steps:
            scheduler.step()

        t1 = time.time()
        dt = t1 - t0

        if step > 5:
            total_training_time += dt

        # Logging
        loss_val = loss.item()

        # Fast fail
        if math.isnan(loss_val) or loss_val > 100:
            print("\nFAIL — loss exploded")
            return None

        ema = 0.95
        smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
        debiased = smooth_loss / (1 - ema ** (step + 1))
        pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
        remaining = max(0, TIME_BUDGET - total_training_time)
        lr_now = optimizer.param_groups[0]["lr"]

        if step % 50 == 0 or step < 5:
            print(f"\r  step {step:05d} ({pct_done:.1f}%) | loss: {debiased:.6f} | lr: {lr_now:.2e} | dt: {dt*1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # Periodic validation
        if step > 0 and step % (steps_per_epoch) == 0:
            metrics = evaluate(model, X_val, y_val, close_val, BATCH_SIZE)
            if metrics["val_mae"] < best_val_mae:
                best_val_mae = metrics["val_mae"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"\n  [eval] val_mae: {metrics['val_mae']:.4f}% | direction_acc: {metrics['direction_accuracy']:.1f}% | best_mae: {best_val_mae:.4f}%")

        # GC management
        if step == 0:
            gc.collect()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        # Time's up
        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    metrics = evaluate(model, X_val, y_val, close_val, BATCH_SIZE)

    # Predict latest price
    device_t = "cuda" if torch.cuda.is_available() else "cpu"
    last_seq = torch.from_numpy(X_val[-1:]).to(device_t)
    with torch.no_grad():
        pred_pct = model(last_seq).item()
    last_close = close_val[-1]
    pred_price = last_close * (1 + pred_pct)
    direction = "UP" if pred_pct > 0 else "DOWN"

    return {
        **metrics,
        "num_steps": step,
        "num_params": num_params,
        "training_seconds": total_training_time,
        "total_seconds": time.time() - t_start,
        "last_close": last_close,
        "predicted_change_pct": pred_pct * 100,
        "predicted_price": pred_price,
        "direction": direction,
        "ticker": ticker_key,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_global = time.time()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    peak_vram = 0.0
    all_results = {}

    for ticker in TICKERS:
        result = train_ticker(ticker)
        if result is not None:
            all_results[ticker] = result

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

    t_total = time.time() - t_global

    # Print summary in autoresearch format
    print("\n---")
    for ticker, r in all_results.items():
        print(f"ticker:              {ticker}")
        print(f"val_mae:             {r['val_mae']:.6f}")
        print(f"val_rmse:            {r['val_rmse']:.6f}")
        print(f"direction_accuracy:  {r['direction_accuracy']:.2f}")
        print(f"last_close:          {r['last_close']:.2f}")
        print(f"predicted_change:    {r['predicted_change_pct']:+.4f}%")
        print(f"predicted_price:     {r['predicted_price']:.2f}")
        print(f"direction:           {r['direction']}")
        print(f"training_seconds:    {r['training_seconds']:.1f}")
        print(f"num_steps:           {r['num_steps']}")
        print(f"num_params:          {r['num_params']}")
        print(f"---")

    print(f"total_seconds:       {t_total:.1f}")
    print(f"peak_vram_mb:        {peak_vram:.1f}")
