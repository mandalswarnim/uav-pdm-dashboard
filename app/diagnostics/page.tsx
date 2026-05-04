'use client';
import dynamic from 'next/dynamic';
import { useEffect } from 'react';
import { useDash } from '@/lib/store';
import StatusBadge from '@/components/HUD/StatusBadge';
import MaintenanceReadout from '@/components/Diagnostics/MaintenanceReadout';
import AttentionHeatmap from '@/components/Diagnostics/AttentionHeatmap';

const WireframeView = dynamic(() => import('@/components/Diagnostics/WireframeView'), { ssr: false });

export default function DiagnosticsPage() {
  const { assets, selectedId, select, details, loadDetail, fleetLoaded } = useDash();
  const sel = assets.find((a) => a.id === selectedId) ?? assets[0];
  const detail = sel ? details[sel.id] ?? null : null;

  useEffect(() => {
    if (sel) loadDetail(sel.id);
  }, [sel, loadDetail]);

  if (!fleetLoaded) {
    return <div className="p-6 text-xs text-hud-dim">// LOADING FLEET ROSTER...</div>;
  }
  if (!sel) {
    return <div className="p-6 text-xs text-hud-dim">// FLEET EMPTY — run `make export` to bake artifacts.</div>;
  }

  // Anomaly is augmented with detail.anomaly when loaded (richer than manifest).
  const assetForView = detail
    ? { ...sel, anomaly: detail.anomaly ?? null }
    : sel;

  return (
    <div className="grid h-full grid-cols-[200px_1fr_360px] gap-3 p-3">
      {/* Asset picker */}
      <aside className="holo-panel flex flex-col">
        <div className="border-b border-hud-cyan/30 px-3 py-2 text-[11px] uppercase tracking-[0.3em] text-hud-cyan">
          // SUBJECT
        </div>
        <div className="flex-1 overflow-y-auto scrollbar-hud">
          {assets.map((a) => (
            <button
              key={a.id}
              onClick={() => select(a.id)}
              className={`block w-full border-b border-hud-cyan/10 px-3 py-2 text-left text-[11px] uppercase tracking-[0.2em] transition-all hover:bg-hud-cyan/5 ${
                a.id === selectedId ? 'bg-hud-cyan/10 text-hud-cyan' : 'text-hud-dim'
              }`}
            >
              <div className="flex justify-between">
                <span className="font-bold">{a.name}</span>
                <span>{a.rul.toFixed(0)}%</span>
              </div>
              <div className="flex justify-between text-[9px] opacity-70">
                <span>{a.id}</span>
                <span>{a.data_source}</span>
              </div>
            </button>
          ))}
        </div>
      </aside>

      {/* Center column: 3D wireframe + attention heatmap stacked */}
      <section className="grid grid-rows-[1fr_240px] gap-3">
        <div className="holo-panel relative">
          <div className="absolute left-3 top-2 z-10 text-[11px] uppercase tracking-[0.3em] text-hud-cyan">
            // DIGITAL TWIN · WIREFRAME · XAI OVERLAY
          </div>
          <div className="absolute right-3 top-2 z-10 flex gap-3 text-[10px] uppercase tracking-[0.2em]">
            <span className="text-hud-cyan">● HEALTHY</span>
            <span className="text-hud-red">● ANOMALY</span>
          </div>
          <WireframeView asset={assetForView} />
        </div>

        <div className="holo-panel p-3">
          <div className="mb-1 flex items-baseline justify-between text-[11px] uppercase tracking-[0.3em] text-hud-cyan">
            <span>// SELF-ATTENTION (TRANSFORMER) · TIME × TIME</span>
            <span className="text-hud-dim">{detail?.attention_2d ? `T = ${detail.attention_2d.length}` : ''}</span>
          </div>
          <div className="h-[180px]">
            <AttentionHeatmap matrix={detail?.attention_2d ?? null} height={180} />
          </div>
        </div>
      </section>

      {/* Right column: dossier */}
      <aside className="holo-panel flex flex-col overflow-y-auto p-4 scrollbar-hud">
        <div className="text-2xl font-bold uppercase tracking-[0.2em] text-hud-cyan glitch-text">{sel.name}</div>
        <div className="text-[11px] uppercase tracking-[0.25em] text-hud-dim">
          {sel.id} · {sel.class} · <span className="text-hud-cyan">{sel.data_source}</span>
        </div>
        <div className="mt-2 mb-4 flex items-center gap-2">
          <StatusBadge status={sel.status} />
          <span className="text-[11px] uppercase tracking-[0.2em] text-hud-dim">RUL {sel.rul.toFixed(1)}%</span>
        </div>
        <MaintenanceReadout detail={detail} />
      </aside>
    </div>
  );
}
