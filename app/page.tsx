import Link from 'next/link';

const tiles = [
  {
    href: '/armory',
    title: 'THE ARMORY',
    sub: 'Fleet Inventory · Asset Selection',
    desc: 'Holographic carousel of all available UAV and missile platforms. Inspect health, RUL, and deploy.',
  },
  {
    href: '/mission',
    title: 'THE MISSION',
    sub: 'Live Deployment · Real-Time Telemetry',
    desc: 'Tactical HUD with live radar sweep, sensor streams, and dynamic Remaining Useful Life forecasting.',
  },
  {
    href: '/diagnostics',
    title: 'THE DIGITAL TWIN',
    sub: 'Post-Flight Diagnostics · XAI Overlay',
    desc: 'Interactive 3D wireframe with anomaly heatmap and prescriptive maintenance recommendations.',
  },
  {
    href: '/lab',
    title: 'THE MODEL LAB',
    sub: 'LSTM × Transformer × 1D-CNN · C-MAPSS + UAV',
    desc: 'Architecture comparison with RMSE, PHM 2008 score, training curves, and per-asset prediction overlays.',
  },
];

export default function Home() {
  return (
    <div className="relative h-full overflow-y-auto px-8 py-10 scrollbar-hud">
      <div className="mx-auto max-w-6xl">
        <div className="holo-panel mb-8 px-6 py-5">
          <div className="text-[11px] uppercase tracking-[0.4em] text-hud-cyan/80">SYSTEM // ARES-PdM v0.1</div>
          <h1 className="mt-1 text-3xl font-bold uppercase tracking-[0.25em] text-hud-cyan glitch-text">
            Tactical UAV Predictive Maintenance
          </h1>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-hud-dim">
            AI-driven Condition-Based Maintenance Plus (CBM+) digital twin. LSTM/Transformer models ingest vibration,
            thermal, and power telemetry to forecast Remaining Useful Life (RUL) across the tactical fleet.
          </p>
        </div>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {tiles.map((t) => (
            <Link key={t.href} href={t.href} className="group">
              <div className="holo-panel h-full p-5 transition-all duration-200 hover:shadow-glow group-hover:-translate-y-1">
                <div className="text-[10px] uppercase tracking-[0.35em] text-hud-cyan/70">{t.sub}</div>
                <div className="mt-2 text-xl font-bold uppercase tracking-[0.18em] text-hud-cyan glitch-text">
                  {t.title}
                </div>
                <p className="mt-3 text-xs leading-relaxed text-hud-dim">{t.desc}</p>
                <div className="mt-6 inline-block btn-hud">ENGAGE →</div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
