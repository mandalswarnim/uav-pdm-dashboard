# ARES PdM — Methodology

> Auto-generated scaffold. Paste sections into your thesis as you write.
> Numerical results in `tables/results.csv` and figures in `figures/` are
> produced by `python -m ml.export` and `python -m ml.figures`.

## 1. Data

### 1.1 NASA C-MAPSS turbofan dataset

We use all four subsets of the C-MAPSS turbofan engine degradation dataset
(Saxena, Goebel, Simon & Eklund, PHM 2008): FD001, FD002, FD003, FD004.
Together these cover the cross product of one or six operating conditions
with one or two fault modes (HPC degradation; HPC + Fan degradation), giving
709 training units and 707 test units across approximately 160 k engine cycles.

Each cycle records 21 sensor channels and three operating-condition variables
(altitude, Mach, throttle resolver angle). Following standard practice we
prune sensors that are constant or near-constant over the training set,
retaining 14 informative channels (s2, s3, s4, s7, s8, s9, s11, s12, s13, s14,
s15, s17, s20, s21).

For multi-condition subsets (FD002, FD004) we cluster the operating-condition
triple into k = 6 clusters with k-means and apply per-cluster z-score
normalization, fitted on the training partition only.

We adopt the **piecewise-linear RUL label** of Heimes (2008) with a ceiling
of 125 cycles: `RUL = clip(end_cycle − cycle, 0, 125)`. This reflects the
operational reality that long-horizon RUL predictions are unreliable while the
asset is still healthy.

We slide a **window of length 30** over each unit to produce
sequence-to-one regression samples; the test partition uses only the most
recent window per unit, matched to its published RUL truth.

### 1.2 Synthesized multirotor UAV flight logs

In addition to the turbofan benchmark, we synthesize a fleet of 20 quadrotor
UAVs flying 10 sorties each (200 flights, ~50k flight-seconds at 10 Hz) using
a procedural model in `ml/data/uav_synth.py`. Each flight follows a
takeoff → climb → cruise → maneuver → descent → land profile and records:

- per-motor vibration RMS, RPM, and ESC temperature (4 motors)
- 6-DOF IMU accelerations and angular rates
- battery voltage, current, capacity-percent, and altitude

Three orthogonal fault modes are injected with severity proportional to
flight-hours-into-life:

| Fault             | Channel signature                                                |
| ----------------- | ---------------------------------------------------------------- |
| Bearing wear      | Slow drift + ~0.7 Hz cyclic component on one motor's vibration   |
| ESC thermal drift | Heat soak: temperature rises with cumulative load                |
| Battery degrade   | Internal-resistance growth → earlier voltage sag under current   |

Drones are randomly assigned a single fault (or healthy, P=0.25) at fleet
generation time. Per-tick RUL labels are computed against a per-drone
nominal end-of-life sampled from `U(140, 220)` flight hours.

The procedural generator is fully reproducible from a seed; we use SEED=1337
throughout.

## 2. Models

We compare three sequence-modeling architectures, each implemented in PyTorch:

1. **LSTM** — 2-layer stacked LSTM (hidden=96), dropout=0.25, MLP head.
2. **Transformer** — 2-layer encoder, d=96, 4 heads, GELU, mean-pool head.
3. **1D-CNN** — 4 stacked 1D-Conv blocks (channels=64, kernel=5) with
   global average pooling (Li et al. 2018 baseline).

For the UAV side, each model also has an auxiliary 4-class fault-classifier
head trained jointly with the RUL regressor (`L = MSE_RUL + 0.4 · CE_fault`).

### 2.1 Training

All runs use AdamW (lr=1e-3, wd=1e-4), batch size 512 (C-MAPSS) or 256 (UAV),
gradient-norm clipping at 5.0, and cosine-annealing learning-rate schedule.
We train for 40 epochs on C-MAPSS and 30 on UAV synth, retaining the
best-validation-RMSE checkpoint.

Train/val splits use a fixed seed (90/10 random). For the UAV dataset we
additionally hold out 20 % of *drones* from the training set so that the
validation distribution contains unseen tail-numbers.

### 2.2 Metrics

For C-MAPSS we report **RMSE** and the asymmetric **PHM 2008 score**

```
score = Σ_i { exp(-d_i / 13) - 1   if d_i < 0
            { exp( d_i / 10) - 1   otherwise        d_i = ŷ_i − y_i
```

which penalizes late predictions (`d_i > 0`) more heavily than early ones,
since predicting failure too late is the safety-critical failure mode in PdM.

For UAV synth we additionally report 4-class **fault-classification accuracy**
on validation drones.

## 3. Explainability

We provide three complementary views of model behaviour:

1. **Self-attention heatmaps** (Transformer only) — per-sample (T × T)
   attention weights from the final encoder layer, visualizing which past
   timesteps the model attends to when predicting current RUL.
2. **Integrated Gradients** (LSTM, CNN) — attributions of the predicted RUL
   w.r.t. each input (time, feature) entry, computed via 32-step IG against
   a zero baseline. Collapsing |IG| over time yields per-sensor importance.
3. **Component-level overlay** — top-attribution sensor → mechanical component
   mapping (e.g., C-MAPSS s11–s14 → HPT/LPT turbine; UAV `vib_rms_m3` →
   rotor-3 bearing). Drives the Diagnostics-page wireframe glow.

## 4. Reproduction

```bash
make setup        # pip install -r requirements.txt
make fetch        # download C-MAPSS to data/cmapss/
make synth        # generate UAV synth fleet to data/uav_synth/
make train        # train all 3 archs × {cmapss, uav}
make export       # bake JSON for the dashboard + thesis tables
make figures      # produce thesis figures
```

End-to-end runtime on Apple M-series with MPS: ~25 min for cmapss, ~10 min
for UAV, ~1 min for export and figures.
