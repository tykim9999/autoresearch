# Stock Price Prediction — NVDA & SK Hynix

Predict stock prices **10 minutes ahead** for NVIDIA (NVDA) and SK Hynix (000660.KS) using deep learning (LSTM / Transformer).

## How It Works

1. **Data**: Fetches real-time and historical intraday data via `yfinance`
2. **Features**: 30+ technical indicators (RSI, MACD, Bollinger Bands, moving averages, momentum, volatility, etc.)
3. **Model**: LSTM with attention mechanism (or Transformer) trained on sliding window sequences
4. **Prediction**: Outputs the predicted price 10 minutes from now

## Quick Start

```bash
# Install dependencies
uv sync

# Train and predict both stocks
uv run predict.py

# Predict NVDA only
uv run predict.py --ticker NVDA

# Predict SK Hynix only
uv run predict.py --ticker HYNIX

# Use Transformer model instead of LSTM
uv run predict.py --model transformer

# Use 1-minute interval data (less history, more granular)
uv run predict.py --interval 1m

# Live prediction mode (refreshes every 60s)
uv run predict.py --ticker NVDA --live
```

## Project Structure

```
predict.py       — Main script: train, evaluate, predict, live mode
model.py         — LSTM (with attention) and Transformer model definitions
data_fetcher.py  — Data fetching, technical indicators, feature engineering
pyproject.toml   — Dependencies
checkpoints/     — Saved model weights (auto-created)
results/         — Prediction plots and live logs (auto-created)
```

## Models

### LSTM with Attention
- 3-layer LSTM with attention mechanism over time steps
- Best for capturing sequential patterns in price movements
- Default model (`--model lstm`)

### Transformer
- Positional encoding + multi-head self-attention
- Alternative architecture (`--model transformer`)

## Features Used

| Category | Features |
|----------|----------|
| Price | Open, High, Low, Close, returns, log returns |
| Moving Averages | SMA(5,10,20,50), EMA(5,10,20), close-to-SMA ratios |
| Momentum | RSI(14), MACD, momentum(1,5,10) |
| Volatility | Bollinger Bands, rolling std(10,20), high-low range |
| Volume | Raw volume, volume SMA(10), volume ratio |

## Output

After training, you get:
- **Console**: Prediction summary with current price, predicted price, direction accuracy
- **Plots**: Saved to `results/` — predicted vs actual, error analysis
- **Checkpoints**: Saved to `checkpoints/` — reusable model weights

## Tickers

| Name | Symbol | Exchange |
|------|--------|----------|
| NVIDIA | NVDA | NASDAQ |
| SK Hynix | 000660.KS | Korea Exchange (KRX) |

## Notes

- Market hours matter: predictions are most useful during active trading
- The model predicts **percentage change**, then converts to price
- Direction accuracy > 55% is considered meaningful for short-term prediction
- Uses Huber loss for robustness to outliers
- Early stopping with patience=10 to prevent overfitting

## License

MIT
