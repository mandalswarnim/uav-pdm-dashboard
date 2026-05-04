"""C-MAPSS preprocessing.

Implements the standard pipeline used in published RUL papers:
  - Load FD00{1..4} train/test/RUL files
  - Compute piecewise-linear RUL labels with a ceiling (Heimes 2008)
  - Drop near-constant sensors; per-condition z-score normalize
  - Build sliding sequence windows for sequence-to-one regression
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from ml.config import CMAPSS, CMAPSS_DIR, SEED


COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')


@dataclass
class CMapssArrays:
    X: np.ndarray   # (N, T, F)
    y: np.ndarray   # (N,)
    units: np.ndarray  # (N,) source unit id (test set)
    sensor_names: list[str]


def _load_subset(subset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    base = CMAPSS_DIR / 'CMaps'
    train = pd.read_csv(base / f'train_{subset}.txt', sep=r'\s+', header=None, names=COLS)
    test = pd.read_csv(base / f'test_{subset}.txt', sep=r'\s+', header=None, names=COLS)
    rul = np.loadtxt(base / f'RUL_{subset}.txt')
    return train, test, rul


def _add_train_rul(df: pd.DataFrame, clip: int) -> pd.DataFrame:
    last = df.groupby('unit')['cycle'].max().rename('cycle_max')
    df = df.merge(last, on='unit')
    df['rul'] = (df['cycle_max'] - df['cycle']).clip(upper=clip)
    return df.drop(columns=['cycle_max'])


def _add_test_rul(df: pd.DataFrame, rul_truth: np.ndarray, clip: int) -> pd.DataFrame:
    # rul_truth[i] is the RUL of unit (i+1) at its last logged cycle in test.
    last = df.groupby('unit')['cycle'].max().rename('cycle_max').reset_index()
    last['rul_at_last'] = rul_truth.astype(float)
    df = df.merge(last, on='unit')
    df['rul'] = (df['rul_at_last'] + (df['cycle_max'] - df['cycle'])).clip(upper=clip)
    return df.drop(columns=['cycle_max', 'rul_at_last'])


def _select_sensors(df: pd.DataFrame, keep: list[int]) -> tuple[pd.DataFrame, list[str]]:
    cols = [f's{i}' for i in keep]
    return df, cols


def _normalize_per_condition(train: pd.DataFrame, test: pd.DataFrame, sensor_cols: list[str]):
    """Z-score per operating-condition cluster (k=6 KMeans on op-settings).

    For FD001/FD003 there is effectively a single condition so this collapses to a global z-score.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    op_cols = ['op1', 'op2', 'op3']
    n_clusters = 6 if train[op_cols].nunique().sum() > 6 else 1
    if n_clusters > 1:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED).fit(train[op_cols])
        train['cond'] = km.predict(train[op_cols])
        test['cond'] = km.predict(test[op_cols])
    else:
        train['cond'] = 0
        test['cond'] = 0

    scalers = {}
    for c in train['cond'].unique():
        sc = StandardScaler().fit(train.loc[train['cond'] == c, sensor_cols])
        scalers[c] = sc

    def _apply(df):
        out = df.copy()
        for c, sc in scalers.items():
            mask = out['cond'] == c
            out.loc[mask, sensor_cols] = sc.transform(out.loc[mask, sensor_cols])
        return out

    return _apply(train), _apply(test)


def _windows(df: pd.DataFrame, sensor_cols: list[str], seq_len: int, mode: Literal['train', 'test']):
    """Sliding-window sequence-to-one extraction.

    Train: every contiguous window of length `seq_len` per unit yields a sample.
    Test:  per-unit, only the *last* window (most recent observation) → matches test RUL labels.
    """
    Xs, ys, us = [], [], []
    for unit, g in df.groupby('unit', sort=False):
        g = g.sort_values('cycle')
        sensors = g[sensor_cols].to_numpy(dtype=np.float32)
        ruls = g['rul'].to_numpy(dtype=np.float32)
        n = len(g)
        if mode == 'train':
            if n < seq_len:
                continue
            for end in range(seq_len, n + 1):
                Xs.append(sensors[end - seq_len:end])
                ys.append(ruls[end - 1])
                us.append(unit)
        else:
            if n < seq_len:
                # left-pad with the earliest value
                pad = np.repeat(sensors[:1], seq_len - n, axis=0)
                window = np.concatenate([pad, sensors], axis=0)
            else:
                window = sensors[-seq_len:]
            Xs.append(window)
            ys.append(ruls[-1])
            us.append(unit)
    return np.stack(Xs), np.array(ys, dtype=np.float32), np.array(us, dtype=np.int64)


def load_cmapss(subset: str) -> tuple[CMapssArrays, CMapssArrays]:
    """Returns (train_arrays, test_arrays) for a single subset."""
    cfg = CMAPSS
    train, test, rul_truth = _load_subset(subset)
    train = _add_train_rul(train, cfg['rul_clip'])
    test = _add_test_rul(test, rul_truth, cfg['rul_clip'])

    _, sensor_cols = _select_sensors(train, cfg['sensors_keep'])
    # Cast to float64 — some sensor channels arrive as int64 and would refuse
    # in-place z-score assignment on pandas 2.x.
    train[sensor_cols] = train[sensor_cols].astype('float64')
    test[sensor_cols] = test[sensor_cols].astype('float64')
    train, test = _normalize_per_condition(train, test, sensor_cols)

    Xtr, ytr, utr = _windows(train, sensor_cols, cfg['sequence_len'], 'train')
    Xte, yte, ute = _windows(test, sensor_cols, cfg['sequence_len'], 'test')

    return (
        CMapssArrays(Xtr, ytr, utr, sensor_cols),
        CMapssArrays(Xte, yte, ute, sensor_cols),
    )


def load_cmapss_all(subsets=SUBSETS) -> tuple[CMapssArrays, CMapssArrays, dict[str, slice]]:
    """Concatenate multiple subsets; return per-subset slices for test arrays."""
    Xs_tr, ys_tr, us_tr = [], [], []
    Xs_te, ys_te, us_te = [], [], []
    spans: dict[str, slice] = {}
    sensor_names = None
    test_cursor = 0
    unit_offset = 0
    for s in subsets:
        tr, te = load_cmapss(s)
        sensor_names = tr.sensor_names
        Xs_tr.append(tr.X); ys_tr.append(tr.y); us_tr.append(tr.units + unit_offset)
        Xs_te.append(te.X); ys_te.append(te.y); us_te.append(te.units + unit_offset)
        spans[s] = slice(test_cursor, test_cursor + len(te.y))
        test_cursor += len(te.y)
        unit_offset += int(max(tr.units.max(), te.units.max())) + 1

    return (
        CMapssArrays(np.concatenate(Xs_tr), np.concatenate(ys_tr), np.concatenate(us_tr), sensor_names),
        CMapssArrays(np.concatenate(Xs_te), np.concatenate(ys_te), np.concatenate(us_te), sensor_names),
        spans,
    )


# ---- torch glue --------------------------------------------------------------

class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loaders(Xtr, ytr, Xte, yte, batch_size: int, val_frac: float = 0.1):
    full = ArrayDataset(Xtr, ytr)
    n_val = int(len(full) * val_frac)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)
    test_ds = ArrayDataset(Xte, yte)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )


# ---- Metrics -----------------------------------------------------------------

def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric scoring function from the original C-MAPSS challenge.

    Penalizes late predictions (pred > true is dangerous) more heavily than early ones.
    Lower is better. (PHM 2008, Saxena et al.)
    """
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(s.sum())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
