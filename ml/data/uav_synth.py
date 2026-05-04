"""Procedural multirotor flight log generator.

Synthesizes plausible PX4-style telemetry for a fleet of quadrotors over their
operational lifetimes, with three injected fault modes that progress with flight
hours:

  - bearing wear         → motor vibration RMS climbs on one rotor
  - esc_thermal          → ESC temperature shows runaway under sustained load
  - battery degradation  → cell voltage sags earlier; internal resistance ↑

Output: one parquet per drone in ``data/uav_synth/`` plus a manifest.csv.

Each row = one 0.1 s tick. Columns:
  drone_id, flight_id, t, phase,
  vib_rms_m1..m4, motor_rpm_m1..m4, esc_temp_m1..m4,
  imu_accel_xyz, imu_gyro_xyz,
  batt_voltage, batt_current, batt_capacity_pct,
  altitude, vbat_drop_under_load,
  fault_class (0 healthy, 1 bearing, 2 esc, 3 battery),
  rul (cumulative-flight-hours-remaining label).

The degradation models are simple but consistent: this is *synthetic* data
designed to expose learnable PdM signals, not to replicate real PX4 dynamics.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ml.config import UAV_DIR, SEED


PHASES = [
    ('takeoff',  6.0,  {'rpm': 0.85, 'vib': 0.6, 'thermal': 0.5}),
    ('climb',    10.0, {'rpm': 0.78, 'vib': 0.5, 'thermal': 0.6}),
    ('cruise',   60.0, {'rpm': 0.55, 'vib': 0.35, 'thermal': 0.5}),
    ('maneuver', 25.0, {'rpm': 0.85, 'vib': 0.7, 'thermal': 0.85}),
    ('descent',  12.0, {'rpm': 0.45, 'vib': 0.3, 'thermal': 0.4}),
    ('land',     5.0,  {'rpm': 0.5, 'vib': 0.4, 'thermal': 0.35}),
]

DT = 0.1                  # seconds per tick → 10 Hz
# Sized so that flights_per_drone × ~0.3 h flights span a meaningful fraction
# of the airframe's life. With EOL=4 h and 10 flights, the last flight starts
# at life_frac ≈ 0.75, giving RUL labels that span ~25..100 across the fleet.
END_OF_LIFE_HOURS = 4.0
FAULT_NAMES = ['healthy', 'bearing', 'esc_thermal', 'battery']


def _phase_profile(total_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (phase_idx_per_tick, load_per_tick). Load is a 0..1 throttle proxy."""
    durations = np.array([d for _, d, _ in PHASES])
    durations = durations * (total_seconds / durations.sum())
    n_ticks = int(total_seconds / DT)
    boundaries = np.cumsum(durations) / DT
    phase_idx = np.zeros(n_ticks, dtype=np.int8)
    load = np.zeros(n_ticks, dtype=np.float32)
    cursor = 0
    for i, (_, _, params) in enumerate(PHASES):
        end = int(boundaries[i])
        phase_idx[cursor:end] = i
        load[cursor:end] = params['rpm']
        cursor = end
    return phase_idx, load


def _simulate_flight(rng: np.random.Generator, hours_into_life: float, fault: int):
    """Generate one full flight. ``fault`` ∈ {0..3}; degradation scales with life."""
    total_seconds = float(rng.uniform(800, 1300))   # ~13–22 min flights
    phase_idx, load = _phase_profile(total_seconds)
    n = len(load)
    t = np.arange(n) * DT

    life_frac = min(hours_into_life / END_OF_LIFE_HOURS, 1.0)

    # ---- Healthy baselines per motor -------------------------------------
    rpm_base = load * 8200 + rng.normal(0, 35, (4, n))             # (4, T)
    vib_base = 0.25 + 0.4 * load + rng.normal(0, 0.04, (4, n))     # g RMS
    esc_base = 28 + 22 * load + rng.normal(0, 0.4, (4, n))         # °C

    # ---- Inject faults (intensity grows with life_frac) ------------------
    fault_motor = int(rng.integers(0, 4))                          # which motor is sick

    if fault == 0:
        # Healthy drones still age uniformly across all components — small
        # global creep on vibration and ESC temperature so the model can
        # estimate RUL from sensors instead of guessing a constant.
        vib_base += life_frac * 0.15
        esc_base += life_frac * 2.0
    elif fault == 1:  # bearing wear
        # Slow drift up + cyclic scrape every rev
        vib_base[fault_motor] += life_frac * (0.6 + 0.25 * np.sin(2 * np.pi * 0.7 * t))
        vib_base[fault_motor] += rng.normal(0, 0.05 + 0.15 * life_frac, n)
        esc_base[fault_motor] += life_frac * 6.0  # mild secondary heating
    elif fault == 2:  # ESC thermal
        # Heat soak — accumulates during sustained-load phases
        cum_load = np.cumsum(load) * DT
        thermal_excursion = life_frac * 0.018 * cum_load
        esc_base[fault_motor] += thermal_excursion + rng.normal(0, 0.6, n)
    elif fault == 3:  # battery degradation
        pass  # handled below in voltage sim

    # ---- Battery model ----------------------------------------------------
    # Voltage sag = f(SOC, current draw, internal_resistance).
    # SOC drops from 100→~25% over the flight; faulty packs sag earlier.
    capacity = 100 - np.cumsum(load) * 100 / load.sum()
    healthy_R = 0.018
    R = healthy_R * (1 + (1.6 * life_frac if fault == 3 else 0.05 * life_frac))
    base_v = 4.05 - (4.05 - 3.30) * (1 - capacity / 100)
    current = 18 * load + rng.normal(0, 0.6, n)                    # amps
    voltage = (base_v * 4) - current * R + rng.normal(0, 0.012, n)
    capacity_pct = capacity

    # ---- IMU body-frame estimates (aggregate from motors) ----------------
    accel_z = -9.81 + (load - 0.5) * 6 + rng.normal(0, 0.18, n)
    accel_x = rng.normal(0, 0.4, n)
    accel_y = rng.normal(0, 0.4, n)
    gyro_x = rng.normal(0, 0.05, n)
    gyro_y = rng.normal(0, 0.05, n)
    gyro_z = rng.normal(0, 0.05, n)

    altitude = np.cumsum((accel_z + 9.81) * DT * DT * 0.5).clip(min=0)

    # ---- Assemble dataframe ----------------------------------------------
    df = pd.DataFrame({
        't': t.astype(np.float32),
        'phase': phase_idx,
        **{f'vib_rms_m{i+1}': vib_base[i].astype(np.float32) for i in range(4)},
        **{f'motor_rpm_m{i+1}': rpm_base[i].astype(np.float32) for i in range(4)},
        **{f'esc_temp_m{i+1}': esc_base[i].astype(np.float32) for i in range(4)},
        'imu_accel_x': accel_x.astype(np.float32),
        'imu_accel_y': accel_y.astype(np.float32),
        'imu_accel_z': accel_z.astype(np.float32),
        'imu_gyro_x': gyro_x.astype(np.float32),
        'imu_gyro_y': gyro_y.astype(np.float32),
        'imu_gyro_z': gyro_z.astype(np.float32),
        'batt_voltage': voltage.astype(np.float32),
        'batt_current': current.astype(np.float32),
        'batt_capacity_pct': capacity_pct.astype(np.float32),
        'altitude': altitude.astype(np.float32),
    })
    return df, total_seconds, fault_motor


def generate_fleet(n_drones: int, flights_per_drone: int, out_dir: Path, seed: int = SEED):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for d in tqdm(range(n_drones), desc='drones'):
        drone_id = f'UAV-{d:03d}'
        # Each drone has one dominant fault mode (or healthy)
        fault = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.30, 0.25, 0.20]))
        # Lifetime in hours is randomized per drone
        nominal_eol = float(rng.uniform(140, 220))

        cum_hours = 0.0
        flight_records = []
        for f in range(flights_per_drone):
            df, secs, fault_motor = _simulate_flight(rng, cum_hours, fault)
            flight_id = f'{drone_id}-F{f:03d}'
            df.insert(0, 'flight_id', flight_id)
            df.insert(0, 'drone_id', drone_id)
            df['fault_class'] = fault
            df['fault_motor'] = fault_motor if fault > 0 else -1
            # RUL tied directly to the same life_frac that scales sensor
            # degradation in `_simulate_flight`. This makes RUL a learnable
            # function of sensors rather than of a hidden per-drone constant.
            tick_hours = df['t'].to_numpy() / 3600.0
            life_frac = np.clip((cum_hours + tick_hours) / END_OF_LIFE_HOURS, 0, 1)
            rul = (1.0 - life_frac) * 100.0
            df['rul'] = rul.astype(np.float32)
            flight_records.append(df)
            cum_hours += secs / 3600.0
            manifest_rows.append({
                'drone_id': drone_id, 'flight_id': flight_id,
                'fault_class': FAULT_NAMES[fault], 'fault_motor': fault_motor if fault > 0 else -1,
                'duration_s': secs, 'cum_hours': cum_hours,
                'rul_at_landing': float(rul[-1]),
                'nominal_eol': float(END_OF_LIFE_HOURS),
            })

        big = pd.concat(flight_records, ignore_index=True)
        big.to_parquet(out_dir / f'{drone_id}.parquet', index=False)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / 'manifest.csv', index=False)
    print(f'✓ Wrote {n_drones} drones × {flights_per_drone} flights → {out_dir}')
    print(f'  Manifest: {out_dir / "manifest.csv"}')
    return manifest


# ---- Loader for ML pipeline --------------------------------------------------

UAV_FEATURES = (
    [f'vib_rms_m{i+1}' for i in range(4)] +
    [f'motor_rpm_m{i+1}' for i in range(4)] +
    [f'esc_temp_m{i+1}' for i in range(4)] +
    ['imu_accel_x', 'imu_accel_y', 'imu_accel_z',
     'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z',
     'batt_voltage', 'batt_current', 'batt_capacity_pct', 'altitude']
)


def load_uav_arrays(seq_len: int, val_drone_frac: float = 0.2):
    """Load synthesized fleet, build sliding windows, return train/val arrays.

    Returns dict with X (N,T,F), y_rul (N,), y_fault (N,), groups by drone.
    """
    from sklearn.preprocessing import StandardScaler

    files = sorted(UAV_DIR.glob('UAV-*.parquet'))
    if not files:
        raise FileNotFoundError(f'No UAV synth data found in {UAV_DIR}. Run uav_synth first.')

    rng = np.random.default_rng(SEED)
    # Stratified hold-out: ensure each fault class appears in both partitions.
    # We probe each drone's parquet for its fault label, then pick at least one
    # drone per class for validation.
    drone_class: dict[str, int] = {}
    for f in files:
        df_head = pd.read_parquet(f, columns=['fault_class']).head(1)
        drone_class[f.stem] = int(df_head['fault_class'].iloc[0])

    by_class: dict[int, list[str]] = {}
    for d, c in drone_class.items():
        by_class.setdefault(c, []).append(d)

    val_drones: set[str] = set()
    for c, ds in by_class.items():
        rng.shuffle(ds)
        # 1 of every class, plus a 20%-of-class top-up rounded up
        n_val_c = max(1, int(round(len(ds) * val_drone_frac)))
        val_drones.update(ds[:n_val_c])

    X_tr, y_rul_tr, y_fault_tr = [], [], []
    X_va, y_rul_va, y_fault_va = [], [], []
    flight_index_va: list[dict] = []

    # Fit a global scaler on training drones first
    train_frames = [pd.read_parquet(f) for f in files if f.stem not in val_drones]
    scaler = StandardScaler().fit(pd.concat(train_frames, ignore_index=True)[UAV_FEATURES])

    for f in files:
        df = pd.read_parquet(f)
        df[UAV_FEATURES] = scaler.transform(df[UAV_FEATURES])
        is_val = f.stem in val_drones
        for flight_id, g in df.groupby('flight_id', sort=False):
            arr = g[UAV_FEATURES].to_numpy(dtype=np.float32)
            ruls = g['rul'].to_numpy(dtype=np.float32)
            fault = int(g['fault_class'].iloc[0])
            n = len(arr)
            if n < seq_len:
                continue
            stride = 5  # downsample to keep dataset tractable
            for end in range(seq_len, n + 1, stride):
                window = arr[end - seq_len:end]
                rul = ruls[end - 1]
                if is_val:
                    X_va.append(window); y_rul_va.append(rul); y_fault_va.append(fault)
                    flight_index_va.append({'drone_id': f.stem, 'flight_id': flight_id, 'tick_end': end})
                else:
                    X_tr.append(window); y_rul_tr.append(rul); y_fault_tr.append(fault)

    return {
        'X_train': np.stack(X_tr), 'y_rul_train': np.array(y_rul_tr, dtype=np.float32),
        'y_fault_train': np.array(y_fault_tr, dtype=np.int64),
        'X_val': np.stack(X_va), 'y_rul_val': np.array(y_rul_va, dtype=np.float32),
        'y_fault_val': np.array(y_fault_va, dtype=np.int64),
        'val_index': flight_index_va,
        'feature_names': UAV_FEATURES,
        'val_drones': sorted(val_drones),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_drones', type=int, default=20)
    p.add_argument('--flights_per_drone', type=int, default=10)
    p.add_argument('--out', type=Path, default=UAV_DIR)
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    generate_fleet(args.n_drones, args.flights_per_drone, args.out, args.seed)


if __name__ == '__main__':
    main()
