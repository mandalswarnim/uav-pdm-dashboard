'use client';
import { useDash } from '@/lib/store';

const fmt = (n: number, d = 0) => n.toFixed(d).padStart(d > 0 ? 5 : 4, '0');

export default function TelemetryPanel() {
  const { current } = useDash();
  const c = current ?? { speed: 0, altitude: 0, thermal: 18, power: 0, vibration: 0, t: 0, rul: 100 };
  const thermalAlert = c.thermal > 78;
  const vibAlert = c.vibration > 1.4;

  return (
    <div className="grid grid-cols-2 gap-2 text-hud-cyan">
      <Cell label="SPEED" value={fmt(c.speed)} unit="KTS" />
      <Cell label="ALTITUDE" value={fmt(c.altitude)} unit="FT" />
      <Cell label="THERMAL" value={fmt(c.thermal, 1)} unit="°C" alert={thermalAlert} />
      <Cell label="POWER" value={fmt(c.power, 1)} unit="A" />
      <Cell label="VIBRATION" value={fmt(c.vibration, 2)} unit="G" alert={vibAlert} />
      <Cell label="MISSION T+" value={fmt(c.t, 1)} unit="S" />
    </div>
  );
}

function Cell({ label, value, unit, alert }: { label: string; value: string; unit: string; alert?: boolean }) {
  return (
    <div className={`border px-3 py-2 ${alert ? 'border-hud-red/70 text-hud-red shadow-glowRed' : 'border-hud-cyan/40'}`}>
      <div className="text-[10px] uppercase tracking-[0.3em] opacity-80">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold tabular-nums">{value}</span>
        <span className="text-[10px] uppercase opacity-70">{unit}</span>
      </div>
    </div>
  );
}
