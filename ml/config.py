"""Centralized config for paths, hyperparameters, and random seeds."""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / 'data'
CMAPSS_DIR = DATA_DIR / 'cmapss'
UAV_DIR = DATA_DIR / 'uav_synth'

ARTIFACTS = ROOT / 'artifacts'
CHECKPOINTS = ARTIFACTS / 'checkpoints'
RUNS = ARTIFACTS / 'runs'             # train/val curves, history JSON

PUBLIC_DATA = ROOT / 'public' / 'data'  # baked JSON the Next.js dashboard reads

THESIS = ROOT / 'thesis'
THESIS_FIG = THESIS / 'figures'
THESIS_TBL = THESIS / 'tables'

for p in [CMAPSS_DIR, UAV_DIR, CHECKPOINTS, RUNS, PUBLIC_DATA, THESIS_FIG, THESIS_TBL]:
    p.mkdir(parents=True, exist_ok=True)


# ---- Hyperparameters ---------------------------------------------------------

SEED = 1337

CMAPSS = dict(
    sequence_len=30,
    rul_clip=125,            # Heimes 2008; standard ceiling for piecewise linear RUL
    sensors_keep=[           # 14 informative sensors after constant-channel pruning
        2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21,
    ],
    batch_size=512,
    epochs=40,
    lr=1e-3,
    val_frac=0.1,            # fraction of *training units* held out for model selection
    patience=8,              # early-stopping patience (epochs without val-RMSE improvement)
)

UAV = dict(
    sequence_len=50,         # 50 ticks @ 10 Hz = 5 seconds
    rul_clip=100,
    sensors_keep=None,       # use all synthesized channels
    batch_size=256,
    epochs=30,
    lr=1e-3,
    val_frac=0.15,           # fraction of *training drones* held out for model selection
    patience=6,
    fault_classes=['healthy', 'bearing', 'esc_thermal', 'battery'],
    # Multi-task loss. RUL MSE is computed on the 0..1 scaled target so the
    # cross-entropy term is comparable in magnitude; a fault label carries no
    # signal at the start of life (a faulty airframe looks healthy at
    # life_frac≈0), so the CE term is down-weighted by (1 - RUL/clip).
    fault_loss_w=0.5,
    fault_ce_life_weight=True,
    fault_ce_min_weight=0.15,
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
