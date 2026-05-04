'use client';
import { useEffect, useRef } from 'react';
import { LiveStream, type FaultName } from './live';
import { useDash } from './store';

/**
 * Subscribes to the FastAPI WebSocket when mode === 'LIVE' && missionRunning.
 * Pipes ticks into the dash store via ingestLiveTick.
 */
export function useLiveStream(opts: {
  fault: FaultName;
  hours: number;
  rate_hz?: number;
  stride?: number;
  seed?: number;
}) {
  const {
    mode, missionRunning,
    ingestLiveTick, setLiveMeta, setLiveStatus,
    resetMission,
  } = useDash();
  const streamRef = useRef<LiveStream | null>(null);

  useEffect(() => {
    if (mode !== 'LIVE' || !missionRunning) return;

    setLiveStatus('CONNECTING');
    const s = new LiveStream(
      { fault: opts.fault, hours: opts.hours, rate_hz: opts.rate_hz, stride: opts.stride, seed: opts.seed },
      {
        onMeta: (m) => { setLiveMeta(m); setLiveStatus('STREAMING'); },
        onTick: (t) => ingestLiveTick(t),
        onEnd:  () => { setLiveStatus('CLOSED'); resetMission(); },
        onError: (msg) => { setLiveStatus('ERROR', msg); },
      },
    );
    s.start();
    streamRef.current = s;
    return () => { s.stop(); streamRef.current = null; };
  }, [mode, missionRunning, opts.fault, opts.hours, opts.rate_hz, opts.stride, opts.seed,
      ingestLiveTick, setLiveMeta, setLiveStatus, resetMission]);
}
