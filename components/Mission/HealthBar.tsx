'use client';
import { useDash } from '@/lib/store';

export default function HealthBar() {
  const { current } = useDash();
  const rul = current?.rul ?? 100;
  const color = rul < 35 ? '#ff2a2a' : rul < 65 ? '#ffb000' : '#00e5ff';
  const label = rul < 35 ? 'CRITICAL' : rul < 65 ? 'DEGRADING' : 'NOMINAL';

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[11px] uppercase tracking-[0.3em]">
        <span className="text-hud-dim">// REMAINING USEFUL LIFE</span>
        <span style={{ color }} className="font-bold">{label}</span>
      </div>
      <div className="relative h-6 w-full border border-hud-cyan/40 bg-black/60">
        <div
          className="h-full transition-all duration-200"
          style={{ width: `${rul}%`, background: color, boxShadow: `0 0 14px ${color}` }}
        />
        {/* Tick marks */}
        <div className="pointer-events-none absolute inset-0 flex">
          {Array.from({ length: 20 }).map((_, i) => (
            <div key={i} className="flex-1 border-r border-black/40" />
          ))}
        </div>
        <div className="absolute inset-0 flex items-center justify-center text-[11px] font-bold tracking-[0.25em] text-black mix-blend-screen">
          <span style={{ color }}>{rul.toFixed(1)} %</span>
        </div>
      </div>
    </div>
  );
}
