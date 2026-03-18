"""
LSTM-based stock price prediction model for 10-minute ahead forecasting.
"""

import torch
import torch.nn as nn
import numpy as np


class StockLSTM(nn.Module):
    """
    Multi-layer LSTM with attention mechanism for stock price prediction.
    Predicts the percentage price change 10 minutes into the future.
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum of LSTM outputs
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden)

        return self.head(context).squeeze(-1)  # (batch,)


class StockTransformer(nn.Module):
    """
    Transformer-based model as an alternative to LSTM.
    """

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        seq_len = x.size(1)
        x = self.input_proj(x) + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x[:, -1, :]  # Use last token
        return self.head(x).squeeze(-1)


def create_sequences(features: np.ndarray, targets: np.ndarray,
                     lookback: int = 60) -> tuple:
    """
    Create sliding window sequences for time series prediction.

    Args:
        features: (num_samples, num_features) array
        targets: (num_samples,) array
        lookback: number of past time steps to include

    Returns:
        X: (num_sequences, lookback, num_features)
        y: (num_sequences,)
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(targets[i])
    return np.array(X), np.array(y)
