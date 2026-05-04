"""Stacked LSTM with optional auxiliary fault-classification head."""
from __future__ import annotations
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    arch_name = 'lstm'

    def __init__(self, input_dim: int, hidden: int = 96, layers: int = 2,
                 dropout: float = 0.25, n_fault_classes: int | None = None):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head_rul = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.head_fault = (
            nn.Linear(hidden, n_fault_classes) if n_fault_classes else None
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        rul = self.head_rul(h_last).squeeze(-1)
        fault = self.head_fault(h_last) if self.head_fault is not None else None
        return rul, fault, {'hidden': h_last}
