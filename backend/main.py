"""FastAPI service for live UAV PdM inference.

Endpoints:
  GET  /healthz              quick liveness probe + which model is loaded
  POST /predict              one-shot prediction on a (T, F) sensor window
  WS   /stream               server-side procedural UAV flight; per-tick RUL +
                              fault-prob frames pushed to the connected client.
                              Query params:
                                fault   ∈ {healthy, bearing, esc_thermal, battery}
                                hours   float, hours-into-life severity (0..200)
                                seed    int, deterministic playback
                                rate_hz int, frames per second (default 10)
                                stride  int, predict every Nth tick (default 5)
"""
from __future__ import annotations
import asyncio
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.inference import get_predictor
from backend.synth_stream import FlightSpec, WindowBuffer, stream_flight
from ml.data.uav_synth import FAULT_NAMES, UAV_FEATURES


app = FastAPI(title='ARES PdM · UAV Live Inference', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ---- Models -----------------------------------------------------------------

class PredictRequest(BaseModel):
    window: list[list[float]]   # (T, F)


class PredictResponse(BaseModel):
    rul: float
    fault_probs: list[float] | None
    fault_classes: list[str]
    arch: str


# ---- Endpoints --------------------------------------------------------------

@app.get('/healthz')
def healthz():
    p = get_predictor()
    return {
        'ok': True,
        'arch': p.arch,
        'device': str(p.device),
        'feature_names': p.feature_names,
        'fault_classes': p.fault_classes,
        'sequence_len': p.seq_len,
    }


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    arr = np.array(req.window, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(400, 'window must be a 2D array of shape (T, F)')
    p = get_predictor()
    try:
        out = p.predict(arr)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return PredictResponse(**out)


@app.websocket('/stream')
async def stream(
    ws: WebSocket,
    fault: str = Query('bearing'),
    hours: float = Query(80.0),
    seed: int = Query(0),
    rate_hz: int = Query(10),
    stride: int = Query(5),
):
    await ws.accept()
    if fault not in FAULT_NAMES:
        await ws.send_json({'type': 'error', 'message': f'unknown fault {fault!r}; one of {FAULT_NAMES}'})
        await ws.close(); return

    p = get_predictor()
    spec = FlightSpec(fault_class=FAULT_NAMES.index(fault), hours_into_life=hours, seed=seed)
    buf = WindowBuffer(maxlen=p.seq_len)
    delay = max(0.0, 1.0 / max(1, rate_hz))

    await ws.send_json({
        'type': 'meta',
        'arch': p.arch,
        'fault_classes': p.fault_classes,
        'feature_names': p.feature_names,
        'seq_len': p.seq_len,
        'spec': {'fault': fault, 'hours': hours, 'seed': seed, 'rate_hz': rate_hz, 'stride': stride},
    })

    try:
        tick = 0
        for frame in stream_flight(spec):
            buf.push(frame['sensor_vector'])
            payload = {
                'type': 'tick',
                't': frame['t'],
                'phase': frame['phase'],
                'sensors': frame['sensors'],
                'fault_motor': frame['fault_motor'],
            }
            # Run inference every `stride` ticks once we have enough history
            if tick % stride == 0 and len(buf) >= p.seq_len:
                pred = p.predict(buf.window())
                payload['prediction'] = pred
            await ws.send_json(payload)
            tick += 1
            await asyncio.sleep(delay)
        await ws.send_json({'type': 'end'})
    except WebSocketDisconnect:
        return
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


@app.on_event('startup')
def _warmup():
    """Force model load on startup so first request isn't slow."""
    p = get_predictor()
    print(f'[backend] ready · arch={p.arch} device={p.device} features={len(p.feature_names)}')
