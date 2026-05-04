"""Lightweight Transformer encoder for sequence-to-one RUL regression.

Exposes attention weights from the final encoder layer for XAI overlays.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _AttnEncoderLayer(nn.Module):
    """Like nn.TransformerEncoderLayer but exposes attention weights."""
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        a, attn_w = self.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1(x + self.dropout(a))
        ff = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x, attn_w  # attn_w: (B, T, T)


class TransformerRegressor(nn.Module):
    arch_name = 'transformer'

    def __init__(self, input_dim: int, d_model: int = 96, nhead: int = 4,
                 layers: int = 2, dim_ff: int = 192, dropout: float = 0.2,
                 n_fault_classes: int | None = None):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = _PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            _AttnEncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(layers)
        ])
        self.head_rul = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.head_fault = (
            nn.Linear(d_model, n_fault_classes) if n_fault_classes else None
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        h = self.pe(self.proj(x))
        attn_layers = []
        for layer in self.layers:
            h, attn = layer(h)
            attn_layers.append(attn)
        # Pool: mean across time
        pooled = h.mean(dim=1)
        rul = self.head_rul(pooled).squeeze(-1)
        fault = self.head_fault(pooled) if self.head_fault is not None else None
        return rul, fault, {'attn': attn_layers[-1], 'hidden': pooled}
