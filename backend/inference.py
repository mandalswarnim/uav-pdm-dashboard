"""Model loading and prediction wrappers for the FastAPI service.

Loads the trained UAV Transformer (preferred) or LSTM/CNN at startup and exposes
a thread-safe predict() that takes a (T, F) sensor window and returns RUL +
per-class fault probabilities.
"""
from __future__ import annotations
from pathlib import Path
import threading

import numpy as np
import torch
import torch.nn.functional as F

from ml.config import CHECKPOINTS, UAV, get_device
from ml.data.uav_synth import UAV_FEATURES, FAULT_NAMES
from ml.models import load_checkpoint

PRIORITY = ['transformer', 'lstm', 'cnn']


class UAVPredictor:
    """Singleton-ish predictor; instantiated once at FastAPI startup."""

    def __init__(self):
        self.device = get_device()
        self.lock = threading.Lock()
        self.arch, self.model, meta = self._load()
        self.feature_names: list[str] = UAV_FEATURES
        self.fault_classes: list[str] = list(FAULT_NAMES)
        self.seq_len: int = int(meta.get('sequence_len', UAV['sequence_len']))
        # Normalisation statistics are stored in the checkpoint at train time
        # (fitted on training drones only) so serving scales live frames
        # exactly as training did. Older checkpoints fall back to refitting.
        if 'scaler_mean' in meta and 'scaler_std' in meta:
            self.scaler_mean = np.asarray(meta['scaler_mean'], dtype=np.float32)
            self.scaler_std = np.asarray(meta['scaler_std'], dtype=np.float32).clip(min=1e-6)
            self.scaler_source = 'checkpoint'
        else:
            print('[backend] ⚠ checkpoint has no scaler stats; refitting from data/uav_synth '
                  '(retrain with `make train-uav` to embed them)')
            self.scaler_mean, self.scaler_std = self._compute_scaler()
            self.scaler_source = 'refit'

    def _load(self) -> tuple[str, torch.nn.Module, dict]:
        for arch in PRIORITY:
            p = CHECKPOINTS / f'uav_{arch}.pt'
            if not p.exists():
                continue
            model, ckpt = load_checkpoint(p, self.device, input_dim=len(UAV_FEATURES),
                                          n_fault_classes=len(FAULT_NAMES))
            print(f'[backend] loaded uav_{arch}.pt on {self.device}')
            return arch, model, ckpt.get('meta', {})
        raise RuntimeError('No UAV checkpoint found. Run `make train-uav` first.')

    def _compute_scaler(self) -> tuple[np.ndarray, np.ndarray]:
        """Legacy fallback: refit a global scaler over the synth fleet on disk."""
        from ml.data.uav_synth import UAV_DIR
        import pandas as pd
        files = sorted(UAV_DIR.glob('UAV-*.parquet'))
        if not files:
            raise RuntimeError(
                'Checkpoint lacks scaler stats and data/uav_synth is empty — predictions '
                'would be unnormalised garbage. Run `make synth && make train-uav`.'
            )
        frames = [pd.read_parquet(f, columns=UAV_FEATURES) for f in files]
        all_data = np.concatenate([f.to_numpy(dtype=np.float32) for f in frames])
        return all_data.mean(0), all_data.std(0).clip(min=1e-6)

    def normalize(self, window: np.ndarray) -> np.ndarray:
        return (window - self.scaler_mean) / self.scaler_std

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> dict:
        """window: (T, F) raw sensor values. Returns RUL + per-class fault probs."""
        if window.shape[-1] != len(self.feature_names):
            raise ValueError(
                f'expected {len(self.feature_names)} features, got {window.shape[-1]}'
            )
        if window.shape[0] < self.seq_len:
            # left-pad with first row so the model gets a fixed-size window
            pad = np.repeat(window[:1], self.seq_len - window.shape[0], axis=0)
            window = np.concatenate([pad, window], axis=0)
        else:
            window = window[-self.seq_len:]

        normed = self.normalize(window)
        x = torch.from_numpy(normed.astype(np.float32)).unsqueeze(0).to(self.device)

        with self.lock:
            rul, fault_logits, _ = self.model(x)

        rul_v = float(rul.item())
        fault_probs = (
            F.softmax(fault_logits, dim=-1).cpu().numpy()[0].tolist()
            if fault_logits is not None else None
        )
        return {
            'rul': rul_v,
            'fault_probs': fault_probs,
            'fault_classes': self.fault_classes,
            'arch': self.arch,
        }


_predictor: UAVPredictor | None = None


def get_predictor() -> UAVPredictor:
    global _predictor
    if _predictor is None:
        _predictor = UAVPredictor()
    return _predictor
