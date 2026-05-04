'use client';
import { useEffect, useState } from 'react';
import { useDash } from '@/lib/store';
import { backendHealth, type FaultName } from '@/lib/live';

const FAULT_OPTIONS: { v: FaultName; label: string }[] = [
  { v: 'healthy',     label: 'HEALTHY' },
  { v: 'bearing',     label: 'BEARING WEAR' },
  { v: 'esc_thermal', label: 'ESC THERMAL' },
  { v: 'battery',     label: 'BATTERY DEGR.' },
];

export interface LiveSpec {
  fault: FaultName;
  hours: number;
  rate_hz: number;
  stride: number;
}

interface Props {
  spec: LiveSpec;
  onSpecChange: (s: LiveSpec) => void;
}

export default function LiveControls({ spec, onSpecChange }: Props) {
  const { liveMeta, liveStatus, liveError, liveFault } = useDash();
  const [health, setHealth] = useState<{ arch: string; device: string } | null>(null);

  useEffect(() => {
    backendHealth().then((h) => h && setHealth({ arch: h.arch, device: h.device }));
  }, []);

  const set = <K extends keyof LiveSpec>(k: K, v: LiveSpec[K]) =>
    onSpecChange({ ...spec, [k]: v });

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between text-[10px] uppercase tracking-[0.3em]">
        <span className="text-hud-cyan">// BACKEND</span>
        {health ? (
          <span className="text-hud-green">● {health.arch.toUpperCase()} · {health.device.toUpperCase()}</span>
        ) : (
          <span className="text-hud-red">● OFFLINE</span>
        )}
      </div>

      <div>
        <Label>FAULT INJECTION</Label>
        <div className="grid grid-cols-2 gap-1">
          {FAULT_OPTIONS.map((o) => (
            <button
              key={o.v}
              onClick={() => set('fault', o.v)}
              className={`border px-2 py-1 text-[10px] uppercase tracking-[0.2em] transition-all ${
                spec.fault === o.v
                  ? 'border-hud-cyan bg-hud-cyan/15 text-hud-cyan shadow-glow'
                  : 'border-hud-cyan/30 text-hud-dim hover:text-hud-cyan'
              }`}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      <Slider label={`HOURS-INTO-LIFE · ${spec.hours.toFixed(2)} h`} min={0} max={4} step={0.05}
              value={spec.hours} onChange={(v) => set('hours', v)} />
      <Slider label={`RATE · ${spec.rate_hz} Hz`} min={5} max={50} step={5}
              value={spec.rate_hz} onChange={(v) => set('rate_hz', v)} />
      <Slider label={`STRIDE · 1 pred / ${spec.stride} ticks`} min={1} max={20} step={1}
              value={spec.stride} onChange={(v) => set('stride', v)} />

      <div>
        <Label>STREAM STATUS</Label>
        <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em]">
          <span className={statusColor(liveStatus)}>● {liveStatus}</span>
          {liveMeta && <span className="text-hud-dim">T={liveMeta.seq_len}</span>}
        </div>
        {liveError && <div className="mt-1 text-[10px] text-hud-red">{liveError}</div>}
      </div>

      {liveFault && (
        <div>
          <Label>LIVE FAULT CLASSIFIER</Label>
          <div className="text-[11px] uppercase tracking-[0.2em] text-hud-amber">
            argmax: <span className="text-hud-cyan font-bold">{liveFault.class}</span>
          </div>
          {liveFault.probs && (
            <div className="mt-1 space-y-1">
              {liveFault.probs.map((p, i) => (
                <FaultBar key={i}
                  label={FAULT_OPTIONS[i]?.label ?? `cls${i}`}
                  v={p}
                  truth={spec.fault === FAULT_OPTIONS[i]?.v}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const statusColor = (s: string) =>
  s === 'STREAMING' ? 'text-hud-green' :
  s === 'CONNECTING' ? 'text-hud-amber' :
  s === 'ERROR' ? 'text-hud-red' :
  'text-hud-dim';

function Label({ children }: { children: React.ReactNode }) {
  return <div className="mb-1 text-[10px] uppercase tracking-[0.3em] text-hud-cyan/80">{children}</div>;
}

function Slider({ label, min, max, step, value, onChange }: {
  label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <div>
      <Label>{label}</Label>
      <input type="range" min={min} max={max} step={step} value={value}
             onChange={(e) => onChange(Number(e.target.value))}
             className="w-full accent-hud-cyan" />
    </div>
  );
}

function FaultBar({ label, v, truth }: { label: string; v: number; truth: boolean }) {
  return (
    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em]">
      <span className={`w-24 ${truth ? 'text-hud-cyan' : 'text-hud-dim'}`}>
        {label}{truth ? ' ★' : ''}
      </span>
      <div className="h-1 flex-1 bg-hud-cyan/10">
        <div className="h-full"
             style={{
               width: `${Math.round(v * 100)}%`,
               background: truth ? '#00ff9c' : '#1e90ff',
               boxShadow: `0 0 6px ${truth ? '#00ff9c' : '#1e90ff'}`,
             }} />
      </div>
      <span className="w-8 text-right text-hud-dim">{(v * 100).toFixed(0)}%</span>
    </div>
  );
}
