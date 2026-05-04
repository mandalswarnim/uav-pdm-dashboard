'use client';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import type { ResultRow } from '@/lib/api';

const PALETTE: Record<string, string> = {
  lstm: '#3b6dd9',
  transformer: '#cc4f4f',
  cnn: '#3a8a4f',
};

interface Props { rows: ResultRow[] }

export default function TrainingCurves({ rows }: Props) {
  const datasets = Array.from(new Set(rows.map((r) => r.dataset)));

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
      {datasets.map((ds) => {
        const subset = rows.filter((r) => r.dataset === ds);
        const maxEp = Math.max(...subset.map((r) => r.history.length));
        // Merge histories into one wide series keyed by epoch
        const merged = Array.from({ length: maxEp }, (_, i) => {
          const e: any = { epoch: i + 1 };
          for (const r of subset) {
            const h = r.history[i];
            if (h) e[r.arch] = h.val_rmse;
          }
          return e;
        });
        return (
          <div key={ds} className="holo-panel p-3">
            <div className="mb-1 text-[11px] uppercase tracking-[0.3em] text-hud-cyan">
              // {ds.toUpperCase()} · VALIDATION RMSE
            </div>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={merged} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                  <XAxis dataKey="epoch" stroke="#1a3a52" tick={{ fill: '#6ea8c4', fontSize: 9 }} />
                  <YAxis stroke="#1a3a52" tick={{ fill: '#6ea8c4', fontSize: 9 }} />
                  <Tooltip
                    contentStyle={{ background: '#02060d', border: '1px solid #00e5ff', fontSize: 11 }}
                    labelStyle={{ color: '#6ea8c4' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 10, color: '#6ea8c4' }} />
                  {subset.map((r) => (
                    <Line
                      key={r.arch}
                      type="monotone"
                      dataKey={r.arch}
                      stroke={PALETTE[r.arch]}
                      dot={false}
                      strokeWidth={1.6}
                      isAnimationActive={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </div>
  );
}
