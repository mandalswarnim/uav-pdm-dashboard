'use client';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useEffect } from 'react';
import { useDash } from '@/lib/store';
import StatusBadge from '@/components/HUD/StatusBadge';

const AssetCarousel = dynamic(() => import('@/components/Armory/AssetCarousel'), { ssr: false });

export default function ArmoryPage() {
  const { assets, selectedId, select, details, loadDetail, fleetLoaded, fleetError } = useDash();
  const sel = assets.find((a) => a.id === selectedId) ?? null;
  const detail = sel ? details[sel.id] ?? null : null;

  useEffect(() => {
    if (sel) loadDetail(sel.id);
  }, [sel, loadDetail]);

  if (fleetError) {
    return <div className="p-6 text-xs text-hud-red">// FLEET LOAD FAILED: {fleetError}</div>;
  }
  if (!fleetLoaded || assets.length === 0) {
    return <div className="p-6 text-xs text-hud-dim">// LOADING FLEET ROSTER...</div>;
  }

  return (
    <div className="grid h-full grid-cols-[280px_1fr_320px] gap-3 p-3">
      {/* Asset list */}
      <aside className="holo-panel flex flex-col">
        <div className="border-b border-hud-cyan/30 px-3 py-2 text-[11px] uppercase tracking-[0.3em] text-hud-cyan">
          // FLEET ROSTER · {assets.length}
        </div>
        <div className="flex-1 overflow-y-auto scrollbar-hud">
          {assets.map((a) => (
            <button
              key={a.id}
              onClick={() => select(a.id)}
              className={`block w-full border-b border-hud-cyan/10 px-3 py-2 text-left transition-all hover:bg-hud-cyan/5 ${
                a.id === selectedId ? 'bg-hud-cyan/10' : ''
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold tracking-[0.18em] text-hud-cyan">{a.name}</span>
                <StatusBadge status={a.status} />
              </div>
              <div className="mt-1 flex justify-between text-[10px] uppercase tracking-[0.2em] text-hud-dim">
                <span>{a.id}</span>
                <span>RUL {a.rul.toFixed(0)}%</span>
              </div>
              <div className="mt-1 flex justify-between text-[9px] uppercase tracking-[0.2em] text-hud-dim/80">
                <span>{a.class}</span>
                <span>{a.data_source}</span>
              </div>
              <div className="mt-1 h-1 w-full bg-hud-cyan/10">
                <div
                  className="h-full transition-all"
                  style={{
                    width: `${Math.min(100, a.rul)}%`,
                    background: a.status === 'CRITICAL' ? '#ff2a2a' : a.status === 'WARNING' ? '#ffb000' : '#00e5ff',
                    boxShadow: `0 0 6px ${a.status === 'CRITICAL' ? '#ff2a2a' : a.status === 'WARNING' ? '#ffb000' : '#00e5ff'}`,
                  }}
                />
              </div>
            </button>
          ))}
        </div>
      </aside>

      {/* 3D carousel */}
      <section className="holo-panel relative">
        <div className="absolute left-3 top-2 z-10 text-[11px] uppercase tracking-[0.3em] text-hud-cyan">
          // HOLOGRAPHIC ARMORY · DRAG TO ROTATE
        </div>
        <div className="absolute right-3 top-2 z-10 flex gap-2">
          <span className="px-2 text-[10px] uppercase tracking-[0.2em] text-hud-cyan">● NOMINAL</span>
          <span className="px-2 text-[10px] uppercase tracking-[0.2em] text-hud-amber">● WARNING</span>
          <span className="px-2 text-[10px] uppercase tracking-[0.2em] text-hud-red">● CRITICAL</span>
        </div>
        <div className="h-full w-full">
          <AssetCarousel />
        </div>
      </section>

      {/* Selected detail */}
      <aside className="holo-panel flex flex-col">
        <div className="border-b border-hud-cyan/30 px-3 py-2 text-[11px] uppercase tracking-[0.3em] text-hud-cyan">
          // DOSSIER
        </div>
        {!sel ? (
          <div className="p-4 text-xs text-hud-dim">Select an asset.</div>
        ) : (
          <div className="flex flex-1 flex-col p-3">
            <div className="text-2xl font-bold uppercase tracking-[0.2em] text-hud-cyan glitch-text">{sel.name}</div>
            <div className="text-[11px] uppercase tracking-[0.25em] text-hud-dim">
              {sel.id} · {sel.class} · <span className="text-hud-cyan">{sel.data_source}</span>
            </div>
            <div className="mt-3"><StatusBadge status={sel.status} /></div>

            <div className="mt-4 space-y-2 text-[11px] uppercase tracking-[0.2em]">
              <Row k="RUL (predicted)"  v={`${sel.rul.toFixed(1)} %`} />
              {detail?.rul_truth != null && <Row k="RUL (truth)" v={`${detail.rul_truth.toFixed(1)} %`} />}
              {detail?.best_arch && <Row k="Best model" v={detail.best_arch.toUpperCase()} />}
              {detail?.subset && <Row k="C-MAPSS subset" v={`${detail.subset} · unit ${detail.unit}`} />}
              {detail?.flight && <Row k="Flight" v={detail.flight.flight_id} />}
              {detail?.anomaly && (
                <>
                  <Row k="Anomaly" v={detail.anomaly.component} alert />
                  <Row k="Severity" v={`${(detail.anomaly.severity * 100).toFixed(0)}%`} alert />
                  <div className="mt-2 border border-hud-red/40 bg-hud-red/5 p-2 text-[10px] leading-relaxed text-hud-red">
                    {detail.anomaly.note}
                  </div>
                </>
              )}
            </div>

            <div className="mt-auto flex flex-col gap-2 pt-4">
              <Link href="/mission" className="btn-hud text-center">DEPLOY → MISSION</Link>
              <Link href="/diagnostics" className="btn-hud text-center">REVIEW → DIGITAL TWIN</Link>
              <Link href="/lab" className="btn-hud text-center">MODEL LAB →</Link>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}

function Row({ k, v, alert }: { k: string; v: string; alert?: boolean }) {
  return (
    <div className={`flex justify-between border-b border-hud-cyan/10 pb-1 ${alert ? 'text-hud-red' : ''}`}>
      <span className="text-hud-dim">{k}</span>
      <span className="font-bold">{v}</span>
    </div>
  );
}
