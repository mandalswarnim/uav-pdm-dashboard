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
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from ml.config import CMAPSS, UAV, SEED, CHECKPOINTS, RUNS, get_device
from ml.data.cmapss import load_cmapss_all, make_loaders, rmse, cmapss_score
from ml.data.uav_synth import load_uav_arrays
from ml.models import ARCHS


@dataclass
class RunMeta:
    dataset: str
    arch: str
    epochs: int
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


def _train_one_epoch(model, loader, opt, device, fault_loss_w=0.0):
    model.train()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    for batch in loader:
        if len(batch) == 3:
            x, y, fc = [b.to(device) for b in batch]
        else:
            x, y = [b.to(device) for b in batch]; fc = None
        opt.zero_grad()
        pred_rul, pred_fault, _ = model(x)
        loss = mse(pred_rul, y)
        if fc is not None and pred_fault is not None and fault_loss_w > 0:
            loss = loss + fault_loss_w * ce(pred_fault, fc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


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
    train_loader, val_loader, test_loader = make_loaders(
        tr.X, tr.y, te.X, te.y, batch_size=cfg['batch_size'],
    )
    meta = {
        'input_dim': tr.X.shape[-1],
        'sequence_len': cfg['sequence_len'],
        'sensor_names': tr.sensor_names,
        'subsets': list(subsets),
        'test_units': te.units.tolist(),
        'test_y': te.y.tolist(),
        'subset_spans': {s: [sl.start, sl.stop] for s, sl in spans.items()},
    }
    return train_loader, val_loader, test_loader, meta, te


def _build_uav():
    cfg = UAV
    print('  Loading UAV synthetic fleet')
    arrs = load_uav_arrays(seq_len=cfg['sequence_len'])
    Xtr = torch.from_numpy(arrs['X_train'])
    ytr = torch.from_numpy(arrs['y_rul_train'])
    ftr = torch.from_numpy(arrs['y_fault_train'])
    Xva = torch.from_numpy(arrs['X_val'])
    yva = torch.from_numpy(arrs['y_rul_val'])
    fva = torch.from_numpy(arrs['y_fault_val'])

    full = TensorDataset(Xtr, ytr, ftr)
    n_val = int(0.1 * len(full))
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)
    test_ds = TensorDataset(Xva, yva, fva)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

    meta = {
        'input_dim': Xtr.shape[-1],
        'sequence_len': cfg['sequence_len'],
        'sensor_names': arrs['feature_names'],
        'fault_classes': cfg['fault_classes'],
        'val_drones': arrs['val_drones'],
        'val_index': arrs['val_index'],
        'test_y': arrs['y_rul_val'].tolist(),
        'test_y_fault': arrs['y_fault_val'].tolist(),
    }
    return train_loader, val_loader, test_loader, meta, arrs


# ---- Driver ------------------------------------------------------------------

def run(dataset: str, arch: str, epochs: int | None = None, subsets=None) -> RunMeta:
    _seed_everything()
    device = get_device()
    print(f'[{dataset}/{arch}] device={device}')

    if dataset == 'cmapss':
        train_loader, val_loader, test_loader, meta, _ = _build_cmapss(subsets or ('FD001',))
        cfg = CMAPSS; n_fault = None; fault_w = 0.0
    elif dataset == 'uav':
        train_loader, val_loader, test_loader, meta, _ = _build_uav()
        cfg = UAV; n_fault = len(cfg['fault_classes']); fault_w = 0.4
    else:
        raise ValueError(dataset)

    epochs = epochs or cfg['epochs']
    Model = ARCHS[arch]
    model = Model(input_dim=meta['input_dim'], n_fault_classes=n_fault).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    best_val = float('inf'); best_state = None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss = _train_one_epoch(model, train_loader, opt, device, fault_loss_w=fault_w)
        yp_v, yt_v, _, _ = _eval(model, val_loader, device)
        val_rmse = rmse(yt_v, yp_v)
        sched.step()
        history.append({'epoch': ep, 'train_loss': tr_loss, 'val_rmse': val_rmse,
                        'lr': sched.get_last_lr()[0]})
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'  ep {ep:02d}/{epochs}  train_loss={tr_loss:.3f}  val_rmse={val_rmse:.3f}')

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
        dataset=dataset, arch=arch, epochs=epochs,
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        test_size=len(test_loader.dataset),
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
