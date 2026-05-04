'use client';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine } from 'recharts';
import { useDash } from '@/lib/store';

interface Props { metric: 'thermal' | 'vibration' | 'power'; color: string; label: string; threshold?: number }

export default function LiveChart({ metric, color, label, threshold }: Props) {
  const { history } = useDash();
  const data = history.map((f) => ({ t: f.t.toFixed(1), v: f[metric] }));

  return (
    <div className="holo-panel h-full p-2">
      <div className="mb-1 flex items-baseline justify-between text-[10px] uppercase tracking-[0.3em] text-hud-dim">
        <span>{label}</span>
        <span style={{ color }}>{data[data.length - 1]?.v.toFixed(2) ?? '—'}</span>
      </div>
      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={data} margin={{ top: 4, right: 4, left: -28, bottom: -8 }}>
          <XAxis dataKey="t" tick={{ fill: '#6ea8c4', fontSize: 9 }} stroke="#1a3a52" />
          <YAxis tick={{ fill: '#6ea8c4', fontSize: 9 }} stroke="#1a3a52" />
          <Tooltip
            contentStyle={{ background: '#02060d', border: '1px solid ' + color, fontSize: 10 }}
            labelStyle={{ color: '#6ea8c4' }}
          />
          {threshold && <ReferenceLine y={threshold} stroke="#ff2a2a" strokeDasharray="3 3" />}
          <Line type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
