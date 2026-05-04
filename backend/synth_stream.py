"""Server-side procedural UAV flight generator for the live WebSocket stream.

Wraps the existing `_simulate_flight` from ml.data.uav_synth into an iterator
that yields per-tick rows. The FastAPI WebSocket consumes this and pushes each
frame (with the model's RUL + fault prediction) to the dashboard.
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ml.data.uav_synth import _simulate_flight, UAV_FEATURES


@dataclass
class FlightSpec:
    fault_class: int = 1          # 0 healthy, 1 bearing, 2 esc, 3 battery
    hours_into_life: float = 80.0  # how worn the airframe is going in
    seed: int = 0


def stream_flight(spec: FlightSpec, max_ticks: int | None = None) -> Iterator[dict]:
    """Yield per-tick telemetry frames at simulated 10 Hz (sleep-paced for live feel)."""
    rng = np.random.default_rng(spec.seed)
    df, total_secs, fault_motor = _simulate_flight(rng, spec.hours_into_life, spec.fault_class)
    n = len(df) if max_ticks is None else min(len(df), max_ticks)

    sensor_arr = df[UAV_FEATURES].to_numpy(dtype=np.float32)
    for i in range(n):
        row = {f: float(df[f].iloc[i]) for f in UAV_FEATURES}
        yield {
            't': float(df['t'].iloc[i]),
            'phase': int(df['phase'].iloc[i]),
            'sensors': row,
            'sensor_vector': sensor_arr[i].tolist(),
            'fault_motor': fault_motor if spec.fault_class > 0 else -1,
        }


class WindowBuffer:
    """Rolling window buffer for the predictor (keeps the last `maxlen` raw frames)."""
    def __init__(self, maxlen: int):
        self.buf: deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, vec: list[float]):
        self.buf.append(np.asarray(vec, dtype=np.float32))

    def window(self) -> np.ndarray:
        return np.stack(list(self.buf))

    def __len__(self): return len(self.buf)
