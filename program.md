# autoresearch — Stock Price Prediction

This is an experiment to have the LLM autonomously improve a stock price prediction model for NVDA and SK Hynix (10-minute ahead forecasting).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar18`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data download, feature engineering, evaluation metrics. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop. Everything is fair game.
4. **Verify data exists**: Check that `~/.cache/stock_prediction/data/` contains pickle files. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains prediction models for both NVDA and SK Hynix. The training script runs for a **fixed time budget of 5 minutes per ticker** (wall clock training time). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture (LSTM, GRU, Transformer, CNN, hybrid), optimizer, hyperparameters, training loop, batch size, model size, feature selection, loss function, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, feature engineering, and training constants (time budget, lookback window, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_mae (validation MAE of predicted % change) and highest direction_accuracy.** The primary metric is `val_mae` — lower is better. `direction_accuracy` is the secondary metric — higher is better (>55% is meaningful). Since the time budget is fixed, you don't need to worry about training time. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the loss function.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_mae gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
ticker:              NVDA
val_mae:             0.123456
val_rmse:            0.234567
direction_accuracy:  54.32
last_close:          125.43
predicted_change:    +0.1234%
predicted_price:     125.58
direction:           UP
training_seconds:    300.1
num_steps:           4523
num_params:          234567
---
ticker:              HYNIX
val_mae:             0.234567
...
---
total_seconds:       625.3
peak_vram_mb:        1234.5
```

You can extract the key metrics from the log file:

```
grep "^val_mae:\|^direction_accuracy:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	nvda_val_mae	hynix_val_mae	nvda_dir_acc	status	description
```

1. git commit hash (short, 7 chars)
2. NVDA val_mae (e.g. 0.1234) — use 0.0000 for crashes
3. HYNIX val_mae (e.g. 0.2345) — use 0.0000 for crashes
4. NVDA direction_accuracy (e.g. 54.3) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar18`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_mae:\|^direction_accuracy:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_mae improved (lower) for either ticker, you "advance" the branch, keeping the git commit
9. If val_mae is equal or worse for both, you git reset back to where you started

**Ideas to try** (in rough priority order):
- Different architectures: GRU, Transformer encoder, 1D-CNN + LSTM hybrid, TCN (Temporal Convolution Network)
- Different loss functions: MSE, MAE, asymmetric loss, directional loss
- Learning rate schedules: warmup + cosine decay, reduce on plateau
- Regularization: different dropout rates, weight decay, batch normalization, layer normalization
- Feature selection: which technical indicators help most?
- Ensemble: train multiple models and average predictions
- Attention mechanisms: multi-head self-attention over time steps
- Residual connections in the model
- Different optimizers: Adam, AdamW, RAdam, SGD with momentum
- Gradient clipping strategies
- Different batch sizes and sequence lengths within LOOKBACK

**Timeout**: Each experiment should take ~10 minutes total (5 min per ticker + startup overhead). If a run exceeds 15 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you.
