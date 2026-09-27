"""Group-aware hold-out split.

Sliding windows cut from the same engine / flight overlap heavily, so a random
split over *windows* puts near-duplicates on both sides and makes validation
RMSE optimistic (and checkpoint selection unreliable). This helper holds out
whole groups (C-MAPSS units, UAV drones) instead.
"""
from __future__ import annotations
import numpy as np


def group_holdout(groups: np.ndarray, val_frac: float, seed: int,
                  strata: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) index arrays that never share a group.

    ``groups``  per-sample group id (any hashable dtype).
    ``strata``  optional per-sample label that is constant within a group; when
                given, at least one group from every stratum is held out.
    """
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq, first_idx = np.unique(groups, return_index=True)

    if strata is None:
        group_strata = {g: 0 for g in uniq}
    else:
        strata = np.asarray(strata)
        group_strata = {g: strata[i] for g, i in zip(uniq, first_idx)}

    val_groups: set = set()
    by_stratum: dict = {}
    for g, s in group_strata.items():
        by_stratum.setdefault(s, []).append(g)
    for s, gs in by_stratum.items():
        gs = list(gs)
        rng.shuffle(gs)
        n_val = max(1, int(round(len(gs) * val_frac))) if len(gs) > 1 else 0
        val_groups.update(gs[:n_val])

    val_mask = np.isin(groups, list(val_groups))
    return np.where(~val_mask)[0], np.where(val_mask)[0]
