"""1D temporal CNN baseline (Li et al. 2018-style) for C-MAPSS RUL.

Treats the sequence as a (channels=F, length=T) signal and applies stacked
1D convs with ReLU, then an MLP head.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CNNRegressor(nn.Module):
    arch_name = 'cnn'

    def __init__(self, input_dim: int, channels: int = 64, n_blocks: int = 4,
                 kernel: int = 5, dropout: float = 0.3,
                 n_fault_classes: int | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        in_c = input_dim
        for i in range(n_blocks):
            layers += [
                nn.Conv1d(in_c, channels, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            ]
            in_c = channels
        self.body = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head_rul = nn.Sequential(
            nn.Linear(channels, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.head_fault = (
            nn.Linear(channels, n_fault_classes) if n_fault_classes else None
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, F) → (B, F, T)
        h = self.body(x.transpose(1, 2))
        h = h.mean(dim=-1)         # global average pool over time
        h = self.dropout(h)
        rul = self.head_rul(h).squeeze(-1)
        fault = self.head_fault(h) if self.head_fault is not None else None
        return rul, fault, {'hidden': h}
