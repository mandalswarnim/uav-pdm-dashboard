"""Bake JSON artifacts the Next.js dashboard reads, plus thesis tables.

For each (dataset, arch) checkpoint:
  - Run model on the test set.
  - For C-MAPSS: pick a curated set of 8 engines spanning health states across
    the four subsets, dump per-engine ground-truth + predicted RUL trajectory
    + per-tick sensor traces + attention/IG XAI map.
  - For UAV: pick 4 representative drones (one per fault class incl. healthy).

Outputs:
  public/data/assets/<asset_id>.json   ← consumed by /armory, /diagnostics, /lab
  public/data/manifest.json            ← roster the dashboard reads on load
  public/data/results.json             ← Model Lab metrics table
  thesis/tables/results.csv, results.tex
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from ml.config import (
    CMAPSS, UAV, CHECKPOINTS, RUNS, PUBLIC_DATA, THESIS_TBL, get_device,
)
from ml.data.cmapss import load_cmapss_all, rmse, cmapss_score
from ml.data.uav_synth import load_uav_arrays, UAV_FEATURES, FAULT_NAMES
from ml.models import ARCHS
from ml.xai import extract_attention, integrated_gradients, sensor_importance


# Curated 12-asset roster: 8 turbofan-backed + 4 quadrotor.
CURATED_TURBOFAN = [
    # (asset_id, name, class, subset, unit_idx_in_subset_test_set)
    ('AGM-09', 'HELLFIRE-IX',  'MISSILE-AGM',    'FD001', 24),
    ('CRZ-04', 'TOMA-IV',      'MISSILE-CRUISE', 'FD003', 17),
    ('RP-007', 'REAPER-VII',   'UAV-STRIKE',     'FD001', 51),
    ('RP-022', 'REAPER-XXII',  'UAV-STRIKE',     'FD002', 33),
    ('RP-031', 'REAPER-XXXI',  'UAV-STRIKE',     'FD004', 12),
    ('CRZ-11', 'TOMA-XI',      'MISSILE-CRUISE', 'FD002',  8),
    ('AGM-21', 'HELLFIRE-XXI', 'MISSILE-AGM',    'FD003', 41),
    ('CRZ-19', 'TOMA-XIX',     'MISSILE-CRUISE', 'FD004', 25),
]
CURATED_UAV = [
    # (asset_id, name, class, prefer_fault)
    ('RV-001', 'RAVEN-I',      'UAV-RECON', 'healthy'),
    ('RV-014', 'RAVEN-XIV',    'UAV-RECON', 'bearing'),
    ('SW-101', 'SWARM-α-101',  'UAV-SWARM', 'esc_thermal'),
    ('SW-118', 'SWARM-α-118',  'UAV-SWARM', 'battery'),
]


@torch.no_grad()
def _batched_predict(model, X, batch=256, with_fault=False):
    model.eval()
    rul_chunks, fault_chunks = [], []
    for i in range(0, len(X), batch):
        chunk = X[i:i + batch]
        r, f, _ = model(chunk)
        rul_chunks.append(r.cpu().numpy())
        if with_fault and f is not None:
            fault_chunks.append(f.argmax(-1).cpu().numpy())
    rul = np.concatenate(rul_chunks)
    if with_fault:
        fault = np.concatenate(fault_chunks) if fault_chunks else None
        return rul, fault
    return rul


def _rul_to_status(rul: float) -> str:
    if rul < 35: return 'CRITICAL'
    if rul < 65: return 'WARNING'
    return 'NOMINAL'


def _load_cmapss_predictions():
    """For each arch, load checkpoint, run on full test, return predictions + extras."""
    device = get_device()
    tr, te, spans = load_cmapss_all(('FD001', 'FD002', 'FD003', 'FD004'))
    out = {}
    for arch in ARCHS:
        ckpt_path = CHECKPOINTS / f'cmapss_{arch}.pt'
        if not ckpt_path.exists():
            print(f'  ⚠ skipping cmapss_{arch} — no checkpoint')
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        Model = ARCHS[arch]
        model = Model(input_dim=tr.X.shape[-1]).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        X = torch.from_numpy(te.X).to(device)
        preds = _batched_predict(model, X, batch=256)
        out[arch] = {
            'preds': preds,
            'model': model,
            'meta': ckpt['meta'],
            'X': X, 'y': te.y, 'units': te.units, 'spans': spans,
            'sensor_names': tr.sensor_names,
        }
    return out


def _bake_cmapss_assets(results):
    """For each curated turbofan asset, write a per-asset JSON with all model preds + XAI."""
    if not results:
        return []
    spans = next(iter(results.values()))['spans']
    sensor_names = next(iter(results.values()))['sensor_names']
    assets_meta = []
    device = get_device()
    seq_len = CMAPSS['sequence_len']

    for asset_id, name, cls, subset, unit_idx in CURATED_TURBOFAN:
        sl = spans[subset]
        # Find the i-th unique unit in that subset's test set.
        units_in_subset = next(iter(results.values()))['units'][sl]
        unique = np.unique(units_in_subset)
        if unit_idx >= len(unique):
            unit_idx = len(unique) - 1
        target_unit = unique[unit_idx]
        # Index inside the global test array
        global_mask = np.zeros(len(next(iter(results.values()))['units']), dtype=bool)
        global_mask[sl] = (units_in_subset == target_unit)
        idx = int(np.where(global_mask)[0][0])

        # Use Transformer attention if available, else IG from CNN/LSTM.
        per_arch = {}
        attention_2d = None
        ig_per_feature = None
        for arch, r in results.items():
            per_arch[arch] = float(r['preds'][idx])
            if arch == 'transformer' and attention_2d is None:
                attn = extract_attention(r['model'], r['X'][idx:idx+1]).cpu().numpy()[0]
                attention_2d = attn.tolist()  # (T, T)
            if arch == 'lstm' and ig_per_feature is None:
                ig = integrated_gradients(r['model'], r['X'][idx:idx+1].clone()).cpu().numpy()[0]
                # collapse across time for the bar chart
                ig_per_feature = np.abs(ig).mean(axis=0)
                ig_per_feature = (ig_per_feature / (ig_per_feature.max() + 1e-9)).tolist()

        # Pick the "best" (lowest-RMSE arch) prediction as headline.
        rmse_per_arch = {arch: rmse(r['y'], r['preds']) for arch, r in results.items()}
        best_arch = min(rmse_per_arch, key=rmse_per_arch.get)
        rul_pred = per_arch[best_arch]
        rul_truth = float(next(iter(results.values()))['y'][idx])
        status = _rul_to_status(rul_pred)

        # Reconstruct the input sensor window for traces.
        sensor_window = next(iter(results.values()))['X'][idx].cpu().numpy().tolist()  # (T, F)

        anomaly = None
        if status != 'NOMINAL' and ig_per_feature:
            top_feat_idx = int(np.argmax(ig_per_feature))
            top_sensor = sensor_names[top_feat_idx]
            comp = _sensor_to_component(top_sensor, cls)
            anomaly = {
                'component': comp,
                'note': f'Top-attribution sensor: {top_sensor}',
                'severity': float(min(1.0, (100 - rul_pred) / 80)),
                'top_sensor_idx': top_feat_idx,
            }

        asset_obj = {
            'id': asset_id,
            'name': name,
            'class': cls,
            'subset': subset,
            'unit': int(target_unit),
            'rul': rul_pred,
            'rul_truth': rul_truth,
            'status': status,
            'predictions_per_arch': per_arch,
            'best_arch': best_arch,
            'rmse_per_arch': rmse_per_arch,
            'sensor_names': sensor_names,
            'sensor_window': sensor_window,
            'attention_2d': attention_2d,
            'sensor_importance': ig_per_feature,
            'anomaly': anomaly,
            'data_source': 'C-MAPSS',
        }
        path = PUBLIC_DATA / 'assets' / f'{asset_id}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asset_obj, indent=2))
        assets_meta.append({
            'id': asset_id, 'name': name, 'class': cls, 'rul': rul_pred,
            'status': status, 'data_source': 'C-MAPSS',
        })
    return assets_meta


def _sensor_to_component(sensor: str, asset_class: str) -> str:
    """Heuristic mapping from sensor name to mechanical component string."""
    # C-MAPSS sensors: see Saxena 2008. We bucket roughly:
    # s2..s4 = LPC/HPC inlet, s7..s9 = HPC outlet, s11..s14 = HPT/LPT, s15..s21 = mechanical
    n = int(sensor.lstrip('s')) if sensor.startswith('s') else -1
    if n in (2, 3, 4):  return 'LPC Stage'
    if n in (7, 8, 9):  return 'HPC Stage'
    if n in (11, 12, 13, 14): return 'HPT / LPT Turbine'
    if n in (15, 17): return 'Combustor'
    if n in (20, 21): return 'Bypass / Bleed'
    return 'Engine Core'


def _bake_uav_assets():
    """For each curated UAV, write per-asset JSON using the UAV models."""
    arrs = load_uav_arrays(seq_len=UAV['sequence_len'])
    val_index = arrs['val_index']
    if not val_index:
        return []

    device = get_device()
    # Prefer transformer (gives attention for free); fall back to LSTM / CNN.
    arch_priority = ['transformer', 'lstm', 'cnn']
    chosen = next((a for a in arch_priority if (CHECKPOINTS / f'uav_{a}.pt').exists()), None)
    if chosen is None:
        print('  ⚠ no UAV checkpoint; skipping UAV asset export')
        return []
    ckpt_path = CHECKPOINTS / f'uav_{chosen}.pt'
    Model = ARCHS[chosen]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Model(input_dim=arrs['X_val'].shape[-1],
                  n_fault_classes=len(UAV['fault_classes'])).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    X = torch.from_numpy(arrs['X_val']).to(device)
    rul_pred, fault_pred = _batched_predict(model, X, batch=256, with_fault=True)

    assets_meta = []
    for asset_id, name, cls, prefer in CURATED_UAV:
        # Find first sample matching the preferred fault class.
        target_fault = FAULT_NAMES.index(prefer)
        candidates = [i for i, f in enumerate(arrs['y_fault_val']) if f == target_fault]
        if not candidates:
            print(f'  ⚠ no UAV val samples for {prefer}, picking any')
            candidates = list(range(min(50, len(arrs['y_fault_val']))))
        idx = candidates[len(candidates) // 2]
        info = val_index[idx]

        # Per-sensor importance from IG
        ig = integrated_gradients(model, X[idx:idx+1].clone()).cpu().numpy()[0]
        feat_importance = np.abs(ig).mean(axis=0)
        feat_importance = (feat_importance / (feat_importance.max() + 1e-9)).tolist()
        attn = extract_attention(model, X[idx:idx+1]).cpu().numpy()[0].tolist()

        rul = float(rul_pred[idx])
        status = _rul_to_status(rul)
        anomaly = None
        if prefer != 'healthy':
            comp_map = {
                'bearing': 'Rotor Bearing',
                'esc_thermal': 'ESC / Power Stage',
                'battery': 'Battery Pack',
            }
            anomaly = {
                'component': comp_map[prefer],
                'note': f'Predicted fault class: {FAULT_NAMES[fault_pred[idx]]}',
                'severity': float(min(1.0, (100 - rul) / 80)),
                'predicted_fault': FAULT_NAMES[int(fault_pred[idx])],
                'true_fault': prefer,
            }

        asset_obj = {
            'id': asset_id, 'name': name, 'class': cls,
            'rul': rul, 'rul_truth': float(arrs['y_rul_val'][idx]),
            'status': status,
            'sensor_names': arrs['feature_names'],
            'sensor_window': X[idx].cpu().numpy().tolist(),
            'attention_2d': attn,
            'sensor_importance': feat_importance,
            'anomaly': anomaly,
            'fault_pred': FAULT_NAMES[int(fault_pred[idx])],
            'fault_true': prefer,
            'flight': info,
            'data_source': 'UAV-Synth',
        }
        path = PUBLIC_DATA / 'assets' / f'{asset_id}.json'
        path.write_text(json.dumps(asset_obj, indent=2))
        assets_meta.append({
            'id': asset_id, 'name': name, 'class': cls, 'rul': rul,
            'status': status, 'data_source': 'UAV-Synth',
        })
    return assets_meta


def _bake_results_table():
    """Compose the Model Lab table from runs/*.json."""
    rows = []
    for f in sorted(RUNS.glob('*.json')):
        d = json.loads(f.read_text())
        rows.append({
            'dataset': d['dataset'], 'arch': d['arch'],
            'rmse': round(d['rmse'], 3),
            'score': None if d.get('score') is None else round(d['score'], 1),
            'fault_acc': None if d.get('fault_acc') is None else round(d['fault_acc'], 3),
            'epochs': d['epochs'],
            'train_size': d['train_size'], 'test_size': d['test_size'],
            'seconds': round(d['seconds'], 1),
            'history': d['history'],
        })
    (PUBLIC_DATA / 'results.json').write_text(json.dumps(rows, indent=2))

    # CSV + LaTeX for thesis
    import csv
    with (THESIS_TBL / 'results.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'arch', 'rmse', 'score', 'fault_acc', 'epochs', 'seconds'])
        for r in rows:
            w.writerow([r['dataset'], r['arch'], r['rmse'], r['score'],
                        r['fault_acc'], r['epochs'], r['seconds']])

    tex = ['\\begin{tabular}{llrrrrr}', '\\toprule',
           'Dataset & Arch & RMSE & Score & Fault Acc & Epochs & Time (s) \\\\', '\\midrule']
    for r in rows:
        tex.append(' & '.join([
            r['dataset'], r['arch'], f"{r['rmse']:.3f}",
            '—' if r['score'] is None else f"{r['score']:.1f}",
            '—' if r['fault_acc'] is None else f"{r['fault_acc']:.3f}",
            str(r['epochs']), f"{r['seconds']:.1f}",
        ]) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    (THESIS_TBL / 'results.tex').write_text('\n'.join(tex))
    return rows


def main():
    print('Baking dashboard artifacts...')
    cmapss_results = _load_cmapss_predictions()
    turbo_assets = _bake_cmapss_assets(cmapss_results)
    uav_assets = _bake_uav_assets()

    manifest = {
        'assets': turbo_assets + uav_assets,
        'generated_at': __import__('datetime').datetime.now(__import__('datetime').UTC).isoformat(),
    }
    (PUBLIC_DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    rows = _bake_results_table()
    print(f'✓ {len(turbo_assets)} turbofan + {len(uav_assets)} UAV assets')
    print(f'✓ {len(rows)} model runs in results table')
    print(f'  → public/data/, thesis/tables/')


if __name__ == '__main__':
    main()
