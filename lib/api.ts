// Static-fetch layer for ML-baked artifacts under /public/data/.
// Phase 3 will add a parallel WebSocket/REST client for live UAV inference.

import type { Asset } from './assets';

export interface AssetDetail extends Asset {
  data_source: 'C-MAPSS' | 'UAV-Synth';
  rul_truth?: number;
  best_arch?: string;
  predictions_per_arch?: Record<string, number>;
  rmse_per_arch?: Record<string, number>;
  sensor_names: string[];
  sensor_window: number[][];           // (T, F)
  attention_2d: number[][] | null;     // (T, T) — Transformer self-attention
  sensor_importance: number[] | null;  // (F,) — IG-derived
  // C-MAPSS-only
  subset?: string;
  unit?: number;
  // UAV-only
  fault_pred?: string;
  fault_true?: string;
  flight?: { drone_id: string; flight_id: string; tick_end: number };
}

export interface Manifest {
  assets: Array<Pick<Asset, 'id' | 'name' | 'class' | 'rul' | 'status'> & {
    data_source: 'C-MAPSS' | 'UAV-Synth';
  }>;
  generated_at: string;
}

export interface ResultRow {
  dataset: 'cmapss' | 'uav';
  arch: 'lstm' | 'transformer' | 'cnn';
  rmse: number;
  score: number | null;
  fault_acc: number | null;
  epochs: number;
  train_size: number;
  test_size: number;
  seconds: number;
  history: { epoch: number; train_loss: number; val_rmse: number; lr: number }[];
}

const BASE = '/data';

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return (await r.json()) as T;
}

export const fetchManifest  = () => getJSON<Manifest>(`${BASE}/manifest.json`);
export const fetchAsset     = (id: string) => getJSON<AssetDetail>(`${BASE}/assets/${id}.json`);
export const fetchResults   = () => getJSON<ResultRow[]>(`${BASE}/results.json`);
