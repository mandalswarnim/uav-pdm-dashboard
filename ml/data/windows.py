"""Memory-lean sliding-window access for long telemetry logs.

Materialising every window of a 10 Hz log as its own (T, F) copy multiplies
memory by ~T/stride (the UAV fleet became ~1.5 GB and swapped a 16 GB machine
into an OOM kill). ``WindowStore`` keeps each scaled tick exactly once and
gathers windows by index at batch time — on the training device, so the
per-batch host→device copy disappears too.
"""
from __future__ import annotations
from typing import Iterator

import numpy as np
import torch


class WindowStore:
    """All ticks of a fleet, concatenated, plus per-tick labels.

    A window is identified by its *end* tick (exclusive) in the concatenated
    array; ``gather(ends)`` returns ``(X, y_rul, y_fault)`` for a batch of ends.
    Windows never cross a flight boundary because callers only generate ends
    from ``seq_len`` onwards within each flight.
    """

    def __init__(self, data: np.ndarray, rul: np.ndarray, fault: np.ndarray, seq_len: int):
        assert data.ndim == 2 and len(data) == len(rul) == len(fault)
        self.seq_len = int(seq_len)
        self.data = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        self.rul = torch.from_numpy(np.ascontiguousarray(rul, dtype=np.float32))
        self.fault = torch.from_numpy(np.ascontiguousarray(fault, dtype=np.int64))
        self._offsets = torch.arange(-self.seq_len, 0)
        self.device = torch.device('cpu')

    @property
    def n_features(self) -> int:
        return int(self.data.shape[1])

    def to(self, device) -> 'WindowStore':
        self.device = torch.device(device)
        self.data = self.data.to(self.device)
        self.rul = self.rul.to(self.device)
        self.fault = self.fault.to(self.device)
        self._offsets = self._offsets.to(self.device)
        return self

    def gather(self, ends: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ends = ends.to(self.device)
        idx = ends.unsqueeze(1) + self._offsets            # (B, T)
        x = self.data[idx]                                  # (B, T, F)
        last = ends - 1
        return x, self.rul[last], self.fault[last]

    def gather_numpy(self, ends: np.ndarray, batch: int = 4096) -> np.ndarray:
        """Materialise (N, T, F) windows on the CPU in chunks (for export / XAI)."""
        ends_t = torch.as_tensor(ends, dtype=torch.long)
        out = []
        for i in range(0, len(ends_t), batch):
            x, _, _ = self.gather(ends_t[i:i + batch])
            out.append(x.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.seq_len, self.n_features), np.float32)


class WindowLoader:
    """DataLoader-like iterator over window ends; yields device-resident batches."""

    def __init__(self, store: WindowStore, ends: np.ndarray, batch_size: int,
                 shuffle: bool, seed: int = 0):
        self.store = store
        self.ends = torch.as_tensor(np.asarray(ends), dtype=torch.long)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self._gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return (len(self.ends) + self.batch_size - 1) // self.batch_size

    @property
    def n_samples(self) -> int:
        return int(len(self.ends))

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        order = torch.randperm(len(self.ends), generator=self._gen) if self.shuffle \
            else torch.arange(len(self.ends))
        ends = self.ends[order]
        for i in range(0, len(ends), self.batch_size):
            yield self.store.gather(ends[i:i + self.batch_size])
