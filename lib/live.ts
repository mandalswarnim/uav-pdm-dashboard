'use client';

// WebSocket client for the FastAPI live UAV inference service.
// Backend at ws://127.0.0.1:8001/stream emits one message per simulated tick.

export type FaultName = 'healthy' | 'bearing' | 'esc_thermal' | 'battery';

export interface LiveMeta {
  type: 'meta';
  arch: string;
  fault_classes: string[];
  feature_names: string[];
  seq_len: number;
  spec: { fault: string; hours: number; seed: number; rate_hz: number; stride: number };
}

export interface LiveTick {
  type: 'tick';
  t: number;
  phase: number;
  sensors: Record<string, number>;
  fault_motor: number;
  prediction?: {
    rul: number;
    fault_probs: number[];
    fault_classes: string[];
    arch: string;
  };
}

export interface LiveEnd  { type: 'end' }
export interface LiveErr  { type: 'error'; message: string }

export type LiveMessage = LiveMeta | LiveTick | LiveEnd | LiveErr;

export interface StreamOptions {
  fault: FaultName;
  hours: number;
  seed?: number;
  rate_hz?: number;
  stride?: number;
  url?: string;
}

export interface StreamHandlers {
  onMeta?: (m: LiveMeta) => void;
  onTick?: (t: LiveTick) => void;
  onEnd?:  () => void;
  onError?: (msg: string) => void;
}

const DEFAULT_URL = 'ws://127.0.0.1:8001/stream';

export class LiveStream {
  private ws: WebSocket | null = null;

  constructor(public opts: StreamOptions, public handlers: StreamHandlers) {}

  start() {
    const params = new URLSearchParams({
      fault:   this.opts.fault,
      hours:   String(this.opts.hours),
      seed:    String(this.opts.seed ?? 0),
      rate_hz: String(this.opts.rate_hz ?? 10),
      stride:  String(this.opts.stride ?? 5),
    });
    const url = `${this.opts.url ?? DEFAULT_URL}?${params}`;
    const ws = new WebSocket(url);
    this.ws = ws;

    ws.onmessage = (ev) => {
      let msg: LiveMessage;
      try { msg = JSON.parse(ev.data); } catch { return; }
      switch (msg.type) {
        case 'meta':  this.handlers.onMeta?.(msg);  break;
        case 'tick':  this.handlers.onTick?.(msg);  break;
        case 'end':   this.handlers.onEnd?.();      break;
        case 'error': this.handlers.onError?.(msg.message); break;
      }
    };
    ws.onerror = () => this.handlers.onError?.('websocket error');
    ws.onclose = () => this.handlers.onEnd?.();
  }

  stop() {
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) {
      this.ws.close();
    }
    this.ws = null;
  }
}

/** Quick health probe. Returns null if backend is unreachable. */
export async function backendHealth(base = 'http://127.0.0.1:8001'): Promise<{
  arch: string; device: string; fault_classes: string[]; feature_names: string[]; sequence_len: number;
} | null> {
  try {
    const r = await fetch(`${base}/healthz`, { cache: 'no-store' });
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}
