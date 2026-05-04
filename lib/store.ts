'use client';
import { create } from 'zustand';
import type { Asset } from './assets';
import { fetchManifest, fetchAsset, type AssetDetail } from './api';
import type { LiveMeta, LiveTick } from './live';

export interface TelemetryFrame {
  t: number;
  speed: number;       // knots
  altitude: number;    // ft
  thermal: number;     // °C
  power: number;       // amps
  vibration: number;   // g
  rul: number;         // 0..100
}

export type MissionMode = 'PROCEDURAL' | 'LIVE';

interface DashState {
  // Fleet roster (loaded from /data/manifest.json on boot)
  assets: Asset[];
  fleetLoaded: boolean;
  fleetError: string | null;
  loadFleet: () => Promise<void>;

  // Selection
  selectedId: string | null;
  select: (id: string | null) => void;

  // Per-asset detail cache
  details: Record<string, AssetDetail>;
  loadDetail: (id: string) => Promise<AssetDetail | null>;

  // Mission simulation (procedural — see useTelemetryStream)
  missionRunning: boolean;
  startMission: () => void;
  stopMission: () => void;
  resetMission: () => void;

  history: TelemetryFrame[];
  current: TelemetryFrame | null;
  pushFrame: (f: TelemetryFrame) => void;

  // Live inference (Phase 3)
  mode: MissionMode;
  setMode: (m: MissionMode) => void;
  liveMeta: LiveMeta | null;
  liveFault: { class: string; probs: number[] | null; arch: string } | null;
  liveStatus: 'IDLE' | 'CONNECTING' | 'STREAMING' | 'CLOSED' | 'ERROR';
  liveError: string | null;
  setLiveMeta: (m: LiveMeta | null) => void;
  setLiveStatus: (s: DashState['liveStatus'], err?: string | null) => void;
  ingestLiveTick: (tick: LiveTick) => void;
}

const initialFrame: TelemetryFrame = {
  t: 0, speed: 0, altitude: 0, thermal: 18, power: 4, vibration: 0.1, rul: 100,
};

export const useDash = create<DashState>((set, get) => ({
  assets: [],
  fleetLoaded: false,
  fleetError: null,
  loadFleet: async () => {
    if (get().fleetLoaded) return;
    try {
      const m = await fetchManifest();
      const assets: Asset[] = m.assets.map((a) => ({ ...a }));
      set({
        assets,
        fleetLoaded: true,
        selectedId: get().selectedId ?? assets[0]?.id ?? null,
      });
    } catch (e) {
      set({ fleetError: (e as Error).message, fleetLoaded: true });
    }
  },

  selectedId: null,
  select: (id) => set({ selectedId: id }),

  details: {},
  loadDetail: async (id) => {
    const cached = get().details[id];
    if (cached) return cached;
    try {
      const d = await fetchAsset(id);
      set((s) => ({ details: { ...s.details, [id]: d } }));
      return d;
    } catch {
      return null;
    }
  },

  missionRunning: false,
  startMission: () => set({ missionRunning: true }),
  stopMission: () => set({ missionRunning: false }),
  resetMission: () => set({
    history: [], current: null, missionRunning: false,
    liveStatus: 'IDLE', liveError: null, liveFault: null,
  }),

  history: [],
  current: null,
  pushFrame: (f) =>
    set((s) => ({
      current: f,
      history: [...s.history.slice(-179), f],
    })),

  mode: 'PROCEDURAL',
  setMode: (m) => set({ mode: m }),
  liveMeta: null,
  liveFault: null,
  liveStatus: 'IDLE',
  liveError: null,
  setLiveMeta: (m) => set({ liveMeta: m }),
  setLiveStatus: (s, err = null) => set({ liveStatus: s, liveError: err }),
  ingestLiveTick: (tick) => {
    const s = tick.sensors;
    // Map raw UAV channels onto the dashboard's abstract telemetry slots.
    const meanRpm = (s.motor_rpm_m1 + s.motor_rpm_m2 + s.motor_rpm_m3 + s.motor_rpm_m4) / 4;
    const maxThermal = Math.max(s.esc_temp_m1, s.esc_temp_m2, s.esc_temp_m3, s.esc_temp_m4);
    const maxVib = Math.max(s.vib_rms_m1, s.vib_rms_m2, s.vib_rms_m3, s.vib_rms_m4);
    const frame: TelemetryFrame = {
      t: tick.t,
      speed: meanRpm * 0.04,         // proxy: ~0.04 kts per RPM unit, scales nicely
      altitude: s.altitude * 3.281,  // m → ft
      thermal: maxThermal,
      power: s.batt_current,
      vibration: maxVib,
      rul: tick.prediction?.rul ?? get().current?.rul ?? 100,
    };
    const fault = tick.prediction
      ? (() => {
          const probs = tick.prediction.fault_probs ?? [];
          const argmax = probs.reduce((mi, v, i, a) => (v > a[mi] ? i : mi), 0);
          return {
            class: tick.prediction.fault_classes[argmax] ?? 'unknown',
            probs,
            arch: tick.prediction.arch,
          };
        })()
      : get().liveFault;
    set((st) => ({
      current: frame,
      history: [...st.history.slice(-179), frame],
      liveFault: fault,
    }));
  },
}));

export { initialFrame };
