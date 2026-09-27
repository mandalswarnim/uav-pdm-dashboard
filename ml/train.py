"""Unified training driver for C-MAPSS and UAV synth datasets.

Usage:
  python -m ml.train --dataset cmapss --subset FD001 FD002 FD003 FD004 \
      --arch lstm transformer cnn
  python -m ml.train --dataset uav --arch lstm transformer cnn

Each (dataset, arch) combination is trained as a separate run. Checkpoints,
training history, and test predictions are written under ``artifacts/``.
"""
from __future__ import annotations
import argparse
import functools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ml.config import CMAPSS, UAV, SEED, CHECKPOINTS, RUNS, get_device
from ml.data.cmapss import load_cmapss_all, make_loaders, rmse, cmapss_score
from ml.data.uav_synth import load_uav_arrays
from ml.data.split import group_holdout
from ml.models import ARCHS
from ml.data.windows import WindowLoader

print = functools.partial(print, flush=True)  # noqa: A001 — readable when piped to a log


@dataclass
class RunMeta:
    dataset: str
    arch: str
    epochs: int              # epochs actually run (≤ max_epochs when early-stopped)
    max_epochs: int
    best_epoch: int
    early_stopped: bool
    split: dict
    train_size: int
    val_size: int
    test_size: int
    rmse: float
    score: float | None
    fault_acc: float | None
    seconds: float
    history: list[dict]


def _seed_everything(seed: int = SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _rul_loss(pred_rul, y, rul_scale):
    """MSE on the 0..1 scaled target so it is commensurate with cross-entropy."""
    return F.mse_loss(pred_rul / rul_scale, y / rul_scale)


def _fault_loss(pred_fault, fc, y, rul_scale, life_weight: bool, min_w: float):
    """Cross-entropy, optionally down-weighted early in life.

    A per-drone fault label is constant over the whole lifetime, but the
    degradation signature is proportional to life_frac = 1 - RUL/clip, so at
    the start of life the label is unlearnable noise. Weighting each sample by
    life_frac (floored at ``min_w``) tells the model to commit to a fault class
    only once the evidence exists.
    """
    ce = F.cross_entropy(pred_fault, fc, reduction='none')
    if not life_weight:
        return ce.mean()
    w = (1.0 - y / rul_scale).clamp(min=min_w, max=1.0)
    return (w * ce).sum() / w.sum()


def _train_one_epoch(model, loader, opt, device, rul_scale: float,
                     fault_loss_w=0.0, life_weight=False, min_w=0.15):
    model.train()
    total = 0.0; n = 0
    for batch in loader:
        if len(batch) == 3:
            x, y, fc = [b.to(device) for b in batch]
        else:
            x, y = [b.to(device) for b in batch]; fc = None
        opt.zero_grad()
        pred_rul, pred_fault, _ = model(x)
        loss = _rul_loss(pred_rul, y, rul_scale)
        if fc is not None and pred_fault is not None and fault_loss_w > 0:
            loss = loss + fault_loss_w * _fault_loss(pred_fault, fc, y, rul_scale, life_weight, min_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item() * len(y); n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    preds, ys, faults_true, faults_pred = [], [], [], []
    for batch in loader:
        if len(batch) == 3:
            x, y, fc = [b.to(device) for b in batch]
        else:
            x, y = [b.to(device) for b in batch]; fc = None
        pred_rul, pred_fault, _ = model(x)
        preds.append(pred_rul.cpu().numpy()); ys.append(y.cpu().numpy())
        if fc is not None and pred_fault is not None:
            faults_pred.append(pred_fault.argmax(-1).cpu().numpy())
            faults_true.append(fc.cpu().numpy())
    yp = np.concatenate(preds); yt = np.concatenate(ys)
    fp = np.concatenate(faults_pred) if faults_pred else None
    ft = np.concatenate(faults_true) if faults_true else None
    return yp, yt, fp, ft


# ---- Dataset adapters --------------------------------------------------------

def _build_cmapss(subsets):
    print(f'  Loading C-MAPSS subsets: {subsets}')
    tr, te, spans = load_cmapss_all(subsets)
    cfg = CMAPSS
    train_loader, val_loader, test_loader, split = make_loaders(
        tr.X, tr.y, te.X, te.y, batch_size=cfg['batch_size'],
        val_frac=cfg['val_frac'], groups=tr.units, seed=SEED,
    )
    meta = {
        'split': split,
        'input_dim': tr.X.shape[-1],
        'sequence_len': cfg['sequence_len'],
        'sensor_names': tr.sensor_names,
        'subsets': list(subsets),
        'test_units': te.units.tolist(),
        'test_y': te.y.tolist(),
        'subset_spans': {s: [sl.start, sl.stop] for s, sl in spans.items()},
    }
    return train_loader, val_loader, test_loader, meta, te


def _build_uav(device):
    cfg = UAV
    print('  Loading UAV synthetic fleet')
    arrs = load_uav_arrays(seq_len=cfg['sequence_len'], materialize_val=False)
    # One copy of every tick lives on the device; batches are gathered by index.
    store = arrs['store'].to(device)
    mb = store.data.numel() * 4 / 2**20
    print(f'  window store: {store.data.shape[0]:,} ticks × {store.n_features} features '
          f'({mb:.0f} MB) on {device}')

    # Hold out whole drones (stratified by fault class) for model selection so
    # no validation window overlaps a training window.
    tr_idx, va_idx = group_holdout(arrs['groups_train'], cfg['val_frac'], SEED,
                                   strata=arrs['y_fault_train'])
    inner_val_drones = sorted(arrs['train_drones'][i] for i in np.unique(arrs['groups_train'][va_idx]))
    split = {
        'strategy': 'group-by-drone',
        'train_drones': int(len(np.unique(arrs['groups_train'][tr_idx]))),
        'val_drones': int(len(inner_val_drones)),
        'inner_val_drone_ids': inner_val_drones,
    }

    bs = cfg['batch_size']
    train_loader = WindowLoader(store, arrs['train_ends'][tr_idx], bs, shuffle=True, seed=SEED)
    val_loader = WindowLoader(store, arrs['train_ends'][va_idx], bs, shuffle=False)
    test_loader = WindowLoader(store, arrs['val_ends'], bs, shuffle=False)

    meta = {
        'input_dim': store.n_features,
        'sequence_len': cfg['sequence_len'],
        'sensor_names': arrs['feature_names'],
        'fault_classes': cfg['fault_classes'],
        'split': split,
        # Normalisation statistics fitted on training drones. The backend and
        # export read these from the checkpoint so serving matches training.
        'scaler_mean': arrs['scaler_mean'].tolist(),
        'scaler_std': arrs['scaler_std'].tolist(),
        'val_drones': arrs['val_drones'],
        'val_index': arrs['val_index'],
        'test_y': arrs['y_rul_val'].tolist(),
        'test_y_fault': arrs['y_fault_val'].tolist(),
    }
    return train_loader, val_loader, test_loader, meta, arrs


def _n_samples(loader) -> int:
    return loader.n_samples if hasattr(loader, 'n_samples') else len(loader.dataset)


# ---- Driver ------------------------------------------------------------------

def run(dataset: str, arch: str, epochs: int | None = None, subsets=None) -> RunMeta:
    _seed_everything()
    device = get_device()
    print(f'[{dataset}/{arch}] device={device}')

    if dataset == 'cmapss':
        train_loader, val_loader, test_loader, meta, _ = _build_cmapss(subsets or ('FD001',))
        cfg = CMAPSS; n_fault = None; fault_w = 0.0; life_w = False; min_w = 0.0
    elif dataset == 'uav':
        train_loader, val_loader, test_loader, meta, _ = _build_uav(device)
        cfg = UAV; n_fault = len(cfg['fault_classes'])
        fault_w = cfg['fault_loss_w']; life_w = cfg['fault_ce_life_weight']; min_w = cfg['fault_ce_min_weight']
    else:
        raise ValueError(dataset)
    print(f"  split: {meta['split']}")

    max_epochs = epochs or cfg['epochs']
    patience = cfg['patience']
    rul_scale = float(cfg['rul_clip'])
    meta['rul_scale'] = rul_scale
    Model = ARCHS[arch]
    model = Model(input_dim=meta['input_dim'], n_fault_classes=n_fault, rul_scale=rul_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    history = []
    best_val = float('inf'); best_state = None; best_epoch = 0
    t0 = time.time()
    ep = 0
    for ep in range(1, max_epochs + 1):
        tr_loss = _train_one_epoch(model, train_loader, opt, device, rul_scale,
                                   fault_loss_w=fault_w, life_weight=life_w, min_w=min_w)
        yp_v, yt_v, fp_v, ft_v = _eval(model, val_loader, device)
        val_rmse = rmse(yt_v, yp_v)
        val_fault_acc = float((fp_v == ft_v).mean()) if fp_v is not None else None
        sched.step()
        history.append({'epoch': ep, 'train_loss': tr_loss, 'val_rmse': val_rmse,
                        'val_fault_acc': val_fault_acc, 'lr': sched.get_last_lr()[0]})
        improved = val_rmse < best_val
        if improved:
            best_val = val_rmse; best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        acc_str = '' if val_fault_acc is None else f'  val_fault_acc={val_fault_acc:.3f}'
        print(f'  ep {ep:02d}/{max_epochs}  train_loss={tr_loss:.4f}  val_rmse={val_rmse:.3f}{acc_str}'
              + ('  *' if improved else ''))
        if ep - best_epoch >= patience:
            print(f'  early stop: no val_rmse improvement for {patience} epochs (best ep {best_epoch})')
            break
    epochs_run = ep
    early_stopped = epochs_run < max_epochs

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0

    yp, yt, fp, ft = _eval(model, test_loader, device)
    test_rmse = rmse(yt, yp)
    score = cmapss_score(yt, yp) if dataset == 'cmapss' else None
    fault_acc = float((fp == ft).mean()) if fp is not None else None
    print(f'  TEST  rmse={test_rmse:.3f}  score={score}  fault_acc={fault_acc}')

    # ---- persist artifacts ----
    tag = f'{dataset}_{arch}'
    ckpt = CHECKPOINTS / f'{tag}.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'arch': arch,
        'dataset': dataset,
        'meta': {**meta, 'predictions': yp.tolist()},
    }, ckpt)
    run_meta = RunMeta(
        dataset=dataset, arch=arch, epochs=epochs_run, max_epochs=max_epochs,
        best_epoch=best_epoch, early_stopped=early_stopped, split=meta['split'],
        train_size=_n_samples(train_loader),
        val_size=_n_samples(val_loader),
        test_size=_n_samples(test_loader),
        rmse=test_rmse, score=score, fault_acc=fault_acc,
        seconds=elapsed, history=history,
    )
    with (RUNS / f'{tag}.json').open('w') as f:
        json.dump(asdict(run_meta), f, indent=2)
    print(f'  ✓ saved → {ckpt}, {RUNS / f"{tag}.json"}')
    return run_meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['cmapss', 'uav'], required=True)
    p.add_argument('--arch', nargs='+', choices=list(ARCHS.keys()), required=True)
    p.add_argument('--subset', nargs='+', default=['FD001', 'FD002', 'FD003', 'FD004'])
    p.add_argument('--epochs', type=int, default=None)
    args = p.parse_args()

    for arch in args.arch:
        run(args.dataset, arch, epochs=args.epochs,
            subsets=tuple(args.subset) if args.dataset == 'cmapss' else None)


if __name__ == '__main__':
    main()
