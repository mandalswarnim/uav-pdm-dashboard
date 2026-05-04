'use client';
import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine, Cell, Legend } from 'recharts';
import { fetchAsset, type AssetDetail } from '@/lib/api';
import { useDash } from '@/lib/store';

const PALETTE: Record<string, string> = {
  lstm: '#3b6dd9',
  transformer: '#cc4f4f',
  cnn: '#3a8a4f',
};

export default function PerAssetOverlay() {
  const { assets, selectedId, select } = useDash();
  const [detail, setDetail] = useState<AssetDetail | null>(null);

  useEffect(() => {
    if (!selectedId) return;
    fetchAsset(selectedId).then(setDetail).catch(() => setDetail(null));
  }, [selectedId]);

  const cmapssAssets = assets.filter((a) => a.data_source === 'C-MAPSS');
  const uavAssets    = assets.filter((a) => a.data_source === 'UAV-Synth');

  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-[260px_1fr]">
      <div className="holo-panel p-2">
        <div className="mb-1 px-1 text-[11px] uppercase tracking-[0.3em] text-hud-cyan">// PICK ASSET</div>
        <Section title="C-MAPSS" items={cmapssAssets} selectedId={selectedId} onPick={select} />
        <Section title="UAV-Synth" items={uavAssets} selectedId={selectedId} onPick={select} />
      </div>

      <div className="holo-panel p-3">
        <div className="mb-1 flex items-baseline justify-between text-[11px] uppercase tracking-[0.3em] text-hud-cyan">
          <span>// PER-ARCHITECTURE PREDICTION</span>
          <span className="text-hud-dim">{detail ? `${detail.name} (${detail.id})` : '—'}</span>
        </div>
        <div className="h-[260px]">
          {detail?.predictions_per_arch ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={Object.entries(detail.predictions_per_arch).map(([arch, value]) => ({
                  arch: arch.toUpperCase(),
                  value,
                  fill: PALETTE[arch] ?? '#999',
                }))}
                margin={{ top: 8, right: 12, left: -12, bottom: 0 }}
              >
                <XAxis dataKey="arch" stroke="#1a3a52" tick={{ fill: '#cfeaff', fontSize: 11 }} />
                <YAxis stroke="#1a3a52" tick={{ fill: '#6ea8c4', fontSize: 9 }} />
                <Tooltip contentStyle={{ background: '#02060d', border: '1px solid #00e5ff', fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 10, color: '#6ea8c4' }} />
                {detail.rul_truth != null && (
                  <ReferenceLine
                    y={detail.rul_truth}
                    stroke="#cfeaff"
                    strokeDasharray="4 3"
                    label={{ value: `truth=${detail.rul_truth.toFixed(1)}`, fill: '#cfeaff', fontSize: 10, position: 'right' }}
                  />
                )}
                <Bar dataKey="value" name="Predicted RUL">
                  {Object.entries(detail.predictions_per_arch).map(([arch], i) => (
                    <Cell key={arch} fill={PALETTE[arch]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : detail ? (
            <div className="flex h-full items-center justify-center text-[11px] uppercase tracking-[0.2em] text-hud-dim">
              // UAV asset · per-arch predictions captured at inference time only
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-[11px] uppercase tracking-[0.2em] text-hud-dim">
              // Loading...
            </div>
          )}
        </div>
        {detail && detail.rmse_per_arch && (
          <div className="mt-2 grid grid-cols-3 gap-2 text-[10px] uppercase tracking-[0.2em]">
            {Object.entries(detail.rmse_per_arch).map(([arch, rmse]) => (
              <div key={arch} className="border border-hud-cyan/30 px-2 py-1">
                <div className="text-hud-dim">{arch} test RMSE</div>
                <div className="font-bold" style={{ color: PALETTE[arch] }}>{rmse.toFixed(2)}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Section({
  title, items, selectedId, onPick,
}: { title: string; items: any[]; selectedId: string | null; onPick: (id: string) => void }) {
  if (items.length === 0) return null;
  return (
    <div className="mb-2">
      <div className="mb-1 px-1 text-[10px] uppercase tracking-[0.3em] text-hud-cyan/70">{title}</div>
      <div className="flex flex-col">
        {items.map((a) => (
          <button
            key={a.id}
            onClick={() => onPick(a.id)}
            className={`flex items-center justify-between border-b border-hud-cyan/10 px-2 py-1 text-left text-[10px] uppercase tracking-[0.2em] transition-all hover:bg-hud-cyan/5 ${
              a.id === selectedId ? 'bg-hud-cyan/10 text-hud-cyan' : 'text-hud-dim'
            }`}
          >
            <span>{a.name}</span><span>{a.rul.toFixed(0)}%</span>
          </button>
        ))}
      </div>
    </div>
  );
}
