"""Explainability utilities.

  - extract_attention(): pulls the last-layer self-attention weights from the
    Transformer for one input batch (B, T, T).
  - integrated_gradients(): vanilla IG over a baseline of zeros for LSTM/CNN.
    Returns per-(time, feature) attribution.
  - sensor_importance(): collapses an attribution map to per-feature scalar
    importance for the dashboard's bar chart.
"""
from __future__ import annotations
import torch


@torch.no_grad()
def extract_attention(model, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    _, _, extras = model(x)
    if 'attn' not in extras:
        return torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=x.device)
    return extras['attn']  # (B, T, T)


def integrated_gradients(model, x: torch.Tensor, steps: int = 32) -> torch.Tensor:
    """Vanilla IG attributions w.r.t. RUL output. Returns (B, T, F)."""
    model.eval()
    baseline = torch.zeros_like(x)
    grads_sum = torch.zeros_like(x)
    for k in range(1, steps + 1):
        alpha = k / steps
        xk = (baseline + alpha * (x - baseline)).requires_grad_(True)
        rul, _, _ = model(xk)
        grad = torch.autograd.grad(rul.sum(), xk)[0]
        grads_sum = grads_sum + grad
    avg = grads_sum / steps
    return ((x - baseline) * avg).detach()


def sensor_importance(attribution: torch.Tensor) -> torch.Tensor:
    """(B, T, F) → (B, F) using mean absolute attribution over time."""
    return attribution.abs().mean(dim=1)
