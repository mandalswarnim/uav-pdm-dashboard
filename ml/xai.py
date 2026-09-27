"""Explainability utilities.

  - extract_attention(): pulls the last-layer self-attention weights from the
    Transformer for one input batch (B, T, T).
  - integrated_gradients(): Captum Integrated Gradients over a zero baseline,
    attributing the scalar RUL output. Returns per-(time, feature) attribution.
  - sensor_importance(): collapses an attribution map to per-feature scalar
    importance for the dashboard's bar chart.
"""
from __future__ import annotations
import torch

from captum.attr import IntegratedGradients as _CaptumIG


@torch.no_grad()
def extract_attention(model, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    _, _, extras = model(x)
    if 'attn' not in extras:
        return torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=x.device)
    return extras['attn']  # (B, T, T)


def integrated_gradients(model, x: torch.Tensor, steps: int = 32) -> torch.Tensor:
    """Integrated Gradients attributions w.r.t. the RUL output, via Captum.

    The model forward returns ``(rul, fault, extras)``; we attribute the scalar
    RUL head against a zero baseline and return per-(time, feature) attributions
    of shape (B, T, F), matching the Sundararajan et al. (2017) formulation.
    """
    model.eval()

    def _rul_only(inp: torch.Tensor) -> torch.Tensor:
        rul, _, _ = model(inp)
        return rul

    ig = _CaptumIG(_rul_only)
    baseline = torch.zeros_like(x)
    attr = ig.attribute(x, baselines=baseline, n_steps=steps)
    return attr.detach()


def sensor_importance(attribution: torch.Tensor) -> torch.Tensor:
    """(B, T, F) → (B, F) using mean absolute attribution over time."""
    return attribution.abs().mean(dim=1)
