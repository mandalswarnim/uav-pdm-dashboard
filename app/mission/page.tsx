'use client';
import dynamic from 'next/dynamic';
import { useState } from 'react';
import { useDash } from '@/lib/store';
import { useTelemetryStream } from '@/lib/telemetry';
import { useLiveStream } from '@/lib/useLiveStream';
import TelemetryPanel from '@/components/Mission/TelemetryPanel';
import HealthBar from '@/components/Mission/HealthBar';
import LiveChart from '@/components/Mission/LiveChart';
import LiveControls, { type LiveSpec } from '@/components/Mission/LiveControls';
import StatusBadge from '@/components/HUD/StatusBadge';

const Radar = dynamic(() => import('@/components/Mission/Radar'), { ssr: false });

export default function MissionPage() {
  const {
    assets, selectedId, mode, setMode,
    missionRunning, startMission, stopMission, resetMission, current,
  } = useDash();
  const sel = assets.find((a) => a.id === selectedId);

  const [liveSpec, setLiveSpec] = useState<LiveSpec>({
    fault: 'bearing', hours: 2.5, rate_hz: 20, stride: 5,
  });

  // Procedural simulator (no-op when mode === 'LIVE')
  useTelemetryStream(250);
  // Live FastAPI WebSocket (no-op unless mode === 'LIVE' && missionRunning)
  useLiveStream(liveSpec);

  return (
    <div className="grid h-full grid-cols-[1fr_360px] gap-3 p-3">
      {/* Left: radar + charts */}
      <section className="grid grid-rows-[1fr_180px] gap-3">
        <div className="holo-panel relative">
          <div className="absolute left-3 top-2 z-10 text-[11px] uppercase tracking-[0.3em] text-hud-cyan">
            // TACTICAL RADAR · BAND-X · 360° SWEEP
          </div>
          <div className="absolute right-3 top-2 z-10 flex gap-3 text-[10px] uppercase tracking-[0.2em]">
            <span className="text-hud-cyan">● ALLY</span>
            <span className="text-hud-green">● UNKNOWN</span>
            <span className="text-hud-red">● HOSTILE</span>
          </div>
          <Radar />
        </div>
        <div className="grid grid-cols-3 gap-3">
          <LiveChart metric="thermal" color="#ffb000" label="THERMAL °C" threshold={mode === 'LIVE' ? 70 : 80} />
          <LiveChart metric="vibration" color="#ff2a2a" label="VIBRATION G" threshold={1.0} />
          <LiveChart metric="power" color="#00e5ff" label="POWER A" />
        </div>
      </section>

      {/* Right: HUD column */}
      <aside className="flex flex-col gap-3 overflow-y-auto scrollbar-hud">
        <div className="holo-panel p-3">
          <div className="text-[10px] uppercase tracking-[0.3em] text-hud-cyan/80">// ACTIVE ASSET</div>
          {sel ? (
            <>
              <div className="mt-1 text-2xl font-bold uppercase tracking-[0.2em] text-hud-cyan glitch-text">{sel.name}</div>
              <div className="text-[11px] uppercase tracking-[0.25em] text-hud-dim">
                {sel.id} · {sel.class} · <span className="text-hud-cyan">{sel.data_source}</span>
              </div>
              <div className="mt-2"><StatusBadge status={sel.status} /></div>
            </>
          ) : <div className="text-xs text-hud-dim">No asset selected.</div>}
        </div>

        <div className="holo-panel p-3">
          <HealthBar />
        </div>

        {/* MODE SELECTOR */}
        <div className="holo-panel p-3">
          <div className="mb-2 text-[10px] uppercase tracking-[0.3em] text-hud-cyan/80">// SIMULATION MODE</div>
          <div className="grid grid-cols-2 gap-1">
            <ModeBtn active={mode === 'PROCEDURAL'} onClick={() => { resetMission(); setMode('PROCEDURAL'); }}>
              PROCEDURAL
            </ModeBtn>
            <ModeBtn active={mode === 'LIVE'} onClick={() => { resetMission(); setMode('LIVE'); }}>
              LIVE INFERENCE
            </ModeBtn>
          </div>
          <div className="mt-2 text-[10px] uppercase tracking-[0.25em] text-hud-dim">
            {mode === 'PROCEDURAL'
              ? 'Frames driven by client-side flight envelope; RUL erodes via heuristic.'
              : 'Frames streamed from FastAPI :8001/stream; RUL + fault probs computed by trained UAV model.'}
          </div>
        </div>

        {mode === 'LIVE' ? (
          <div className="holo-panel p-3">
            <div className="mb-2 text-[10px] uppercase tracking-[0.3em] text-hud-cyan/80">// LIVE STREAM CONTROLS</div>
            <LiveControls spec={liveSpec} onSpecChange={setLiveSpec} />
          </div>
        ) : (
          <div className="holo-panel p-3">
            <div className="mb-2 text-[10px] uppercase tracking-[0.3em] text-hud-cyan/80">// LIVE TELEMETRY STREAM</div>
            <TelemetryPanel />
          </div>
        )}

        <div className="holo-panel p-3">
          <div className="mb-2 text-[10px] uppercase tracking-[0.3em] text-hud-cyan/80">// MISSION CONTROL</div>
          <div className="flex flex-wrap gap-2">
            {!missionRunning ? (
              <button onClick={startMission} className="btn-hud">▶ DEPLOY</button>
            ) : (
              <button onClick={stopMission} className="btn-hud" style={{ borderColor: '#ffb000', color: '#ffb000' }}>■ HOLD</button>
            )}
            <button onClick={resetMission} className="btn-hud">↺ RESET</button>
          </div>
        </div>

        {current && current.rul < 50 && (
          <div className="holo-panel border-hud-red/60 p-3 shadow-glowRed">
            <div className="text-[10px] uppercase tracking-[0.3em] text-hud-red">// AI ADVISORY</div>
            <p className="mt-1 text-xs leading-relaxed text-hud-red">
              {mode === 'LIVE' ? 'Trained UAV model' : 'Heuristic envelope'} reports accelerated wear.
              RUL projected below 50%. Recommend RTB and CBM+ inspection.
            </p>
          </div>
        )}
      </aside>
    </div>
  );
}

function ModeBtn({ children, active, onClick }: { children: React.ReactNode; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`border px-2 py-1.5 text-[11px] uppercase tracking-[0.25em] transition-all ${
        active
          ? 'border-hud-cyan bg-hud-cyan/15 text-hud-cyan shadow-glow'
          : 'border-hud-cyan/30 text-hud-dim hover:text-hud-cyan'
      }`}
    >
      {children}
    </button>
  );
}
