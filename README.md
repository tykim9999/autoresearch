# autoresearch — Stock Price Prediction

Autonomous AI-driven experimentation for stock price prediction. An AI agent iteratively modifies the model architecture, hyperparameters, and training loop to find the best 10-minute ahead predictor for **NVIDIA (NVDA)** and **SK Hynix (000660.KS)**.

Built on the [autoresearch](https://github.com/karpathy/autoresearch) framework by @karpathy — same loop, different domain: instead of optimizing LLM pretraining, we optimize stock price prediction.

## How It Works

The repo has three files that matter:

- **`prepare.py`** — fixed: data download (via yfinance), technical feature engineering (30+ indicators), evaluation metrics. Not modified.
- **`train.py`** — the single file the agent edits. Model architecture (LSTM/GRU/Transformer/etc.), optimizer, hyperparameters, training loop. **This file is edited and iterated on by the agent**.
- **`program.md`** — instructions for the autonomous agent. **This file is edited by the human**.

The agent runs an infinite loop: modify `train.py` → train for 5 minutes → check if `val_mae` improved → keep or discard → repeat. You wake up to a log of experiments and (hopefully) a better model.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Download stock data (one-time)
uv run prepare.py

# 3. Run a single training experiment (~10 min, both tickers)
uv run train.py
```

## Running the Agent

Spin up Claude Code (or any coding agent) in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The agent will autonomously experiment with different model architectures and hyperparameters to minimize prediction error.

## Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| `val_mae` | Validation MAE of predicted % change | Lower is better (primary) |
| `direction_accuracy` | % of correct up/down predictions | Higher is better (>55% meaningful) |
| `val_rmse` | Validation RMSE of predicted % change | Lower is better |

## Tickers

| Name | Symbol | Exchange |
|------|--------|----------|
| NVIDIA | NVDA | NASDAQ |
| SK Hynix | 000660.KS | Korea Exchange (KRX) |

## Features Used

30+ technical indicators computed in `prepare.py`:

- **Price**: Open, High, Low, Close, returns
- **Moving Averages**: SMA(5,10,20,50), EMA(5,10,20), close-to-SMA ratios
- **Momentum**: RSI(14), MACD + signal + histogram, momentum(1,5,10)
- **Volatility**: Bollinger Bands + position, rolling std(10,20), high-low range
- **Volume**: Raw volume, volume SMA(10), volume ratio

## Project Structure

```
prepare.py      — data download, features, evaluation (do not modify)
train.py        — model + training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design Choices

- **Single file to modify.** The agent only touches `train.py`. Diffs are reviewable.
- **Fixed time budget.** Training always runs for 5 minutes per ticker. Experiments are directly comparable.
- **Fixed evaluation.** `prepare.py` contains the ground truth metrics. No gaming the eval.
- **Self-contained.** PyTorch + yfinance + scikit-learn. One GPU, one file, one metric.

## License

MIT
