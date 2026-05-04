"""Publication-quality matplotlib figures for the thesis report.

Reads training history + test predictions from artifacts/runs and checkpoints
and writes PNG + PDF into thesis/figures/. All figures use a single style:
  - serif font, 9pt
  - figure widths sized for a 6.5"-text body
  - viridis / muted palette (no pure-RGB)

Generated figures:
  fig_loss_curves.[png|pdf]      — train/val curves, all 6 model variants
  fig_rul_overlay_<asset>.png    — ground-truth vs. predicted RUL per asset
  fig_attention_heatmap_<asset>  — Transformer attention (T×T) for one asset
  fig_sensor_importance.png      — bar chart of mean |IG| across UAV assets
  fig_results_bars.png           — RMSE / Score bars per (dataset, arch)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ml.config import RUNS, CHECKPOINTS, PUBLIC_DATA, THESIS_FIG


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'figure.dpi': 130,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def _save(fig, name: str):
    THESIS_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(THESIS_FIG / f'{name}.{ext}')
    plt.close(fig)


def fig_loss_curves():
    runs = sorted(RUNS.glob('*.json'))
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6), sharey=False)
    palette = {'lstm': '#3b6dd9', 'transformer': '#cc4f4f', 'cnn': '#3a8a4f'}
    for ax, ds in zip(axes, ['cmapss', 'uav']):
        for f in runs:
            d = json.loads(f.read_text())
            if d['dataset'] != ds: continue
            xs = [h['epoch'] for h in d['history']]
            ys = [h['val_rmse'] for h in d['history']]
            ax.plot(xs, ys, label=d['arch'], color=palette.get(d['arch'], '#666'), lw=1.2)
        ax.set_title(ds.upper())
        ax.set_xlabel('epoch'); ax.set_ylabel('val RMSE')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, ls=':', lw=0.4, alpha=0.6)
    fig.tight_layout()
    _save(fig, 'fig_loss_curves')


def fig_rul_overlay():
    assets_dir = PUBLIC_DATA / 'assets'
    if not assets_dir.exists(): return
    files = sorted(assets_dir.glob('*.json'))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.5), sharey=True)
    for ax, f in zip(axes.flat, files):
        a = json.loads(f.read_text())
        preds = a.get('predictions_per_arch') or {}
        if not preds:
            ax.set_visible(False); continue
        truth = a.get('rul_truth')
        archs = list(preds.keys())
        vals = [preds[k] for k in archs]
        bars = ax.bar(archs, vals, color=['#3b6dd9', '#cc4f4f', '#3a8a4f'][:len(archs)])
        if truth is not None:
            ax.axhline(truth, color='black', ls='--', lw=1, label=f'truth={truth:.0f}')
            ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"{a['name']} ({a.get('subset', a.get('data_source'))})", fontsize=8)
        ax.set_ylabel('RUL')
    fig.suptitle('Per-asset RUL prediction across architectures', y=1.02)
    fig.tight_layout()
    _save(fig, 'fig_rul_overlay')


def fig_attention_heatmap():
    assets_dir = PUBLIC_DATA / 'assets'
    if not assets_dir.exists(): return
    candidates = [f for f in assets_dir.glob('*.json')
                  if json.loads(f.read_text()).get('attention_2d')]
    if not candidates: return
    f = candidates[0]
    a = json.loads(f.read_text())
    attn = np.array(a['attention_2d'])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    ax.set_xlabel('attended timestep'); ax.set_ylabel('query timestep')
    ax.set_title(f"Self-attention · {a['name']}")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    _save(fig, f"fig_attention_{a['id']}")


def fig_sensor_importance():
    assets_dir = PUBLIC_DATA / 'assets'
    if not assets_dir.exists(): return
    rows = []
    for f in assets_dir.glob('*.json'):
        a = json.loads(f.read_text())
        if a.get('data_source') != 'UAV-Synth' or not a.get('sensor_importance'): continue
        rows.append((a['name'], a['sensor_names'], a['sensor_importance']))
    if not rows: return
    names = rows[0][1]
    importance = np.mean([r[2] for r in rows], axis=0)
    order = np.argsort(importance)[::-1][:15]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(order)), importance[order], color='#3b6dd9')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[i] for i in order], rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('|IG| (normalized)')
    ax.set_title('Mean sensor attribution across UAV assets')
    fig.tight_layout()
    _save(fig, 'fig_sensor_importance')


def fig_results_bars():
    p = PUBLIC_DATA / 'results.json'
    if not p.exists(): return
    rows = json.loads(p.read_text())
    if not rows: return
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    palette = {'lstm': '#3b6dd9', 'transformer': '#cc4f4f', 'cnn': '#3a8a4f'}
    for ax, ds in zip(axes, ['cmapss', 'uav']):
        sub = [r for r in rows if r['dataset'] == ds]
        if not sub: ax.set_visible(False); continue
        archs = [r['arch'] for r in sub]
        vals = [r['rmse'] for r in sub]
        ax.bar(archs, vals, color=[palette[a] for a in archs])
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'{ds.upper()} · Test RMSE')
        ax.set_ylabel('RMSE')
    fig.tight_layout()
    _save(fig, 'fig_results_bars')


def main():
    print('Generating thesis figures...')
    fig_loss_curves()
    fig_rul_overlay()
    fig_attention_heatmap()
    fig_sensor_importance()
    fig_results_bars()
    print(f'✓ Figures written to {THESIS_FIG}')


if __name__ == '__main__':
    main()
