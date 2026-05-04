'use client';
import { useEffect, useRef } from 'react';
import { useDash, initialFrame, type TelemetryFrame } from './store';

/**
 * Mock telemetry generator — simulates a flight envelope with random
 * thermal/vibration spikes that erode the Remaining Useful Life curve.
 * In production this would be a WebSocket subscription instead.
 */
export function useTelemetryStream(intervalMs = 250) {
  const { missionRunning, pushFrame, current, resetMission, mode } = useDash();
  const lastRef = useRef<TelemetryFrame>(initialFrame);

  useEffect(() => {
    if (!missionRunning || mode !== 'PROCEDURAL') return;
    if (!current) lastRef.current = initialFrame;
    else lastRef.current = current;

    const id = setInterval(() => {
      const prev = lastRef.current;
      const t = prev.t + intervalMs / 1000;

      // Flight phases driven by t
      const phase = t < 12 ? 'climb' : t < 60 ? 'cruise' : t < 80 ? 'maneuver' : 'descent';

      let speed = prev.speed;
      let altitude = prev.altitude;
      let thermal = prev.thermal;
      let power = prev.power;
      let vibration = prev.vibration;

      const jitter = (s: number) => (Math.random() - 0.5) * s;

      if (phase === 'climb') {
        speed += 6 + jitter(2);
        altitude += 220 + jitter(40);
        thermal += 0.6 + jitter(0.2);
        power = 18 + jitter(2);
        vibration = 0.4 + jitter(0.1);
      } else if (phase === 'cruise') {
        speed = 280 + jitter(8);
        altitude += jitter(30);
        thermal += 0.05 + jitter(0.15);
        power = 9 + jitter(1.2);
        vibration = 0.5 + jitter(0.1);
      } else if (phase === 'maneuver') {
        speed = 320 + jitter(20);
        altitude += jitter(120);
        thermal += 0.8 + jitter(0.4);
        power = 22 + jitter(3);
        vibration = 1.4 + jitter(0.6);
      } else {
        speed = Math.max(140, speed - 4 + jitter(3));
        altitude = Math.max(0, altitude - 280 + jitter(50));
        thermal = Math.max(20, thermal - 0.4 + jitter(0.2));
        power = 11 + jitter(1.5);
        vibration = 0.7 + jitter(0.2);
      }

      // Random anomaly spike (~3% per tick during maneuver)
      if (phase === 'maneuver' && Math.random() < 0.03) {
        thermal += 8;
        vibration += 1.6;
      }

      // RUL erosion model — accelerated by thermal & vibration excursions.
      const thermalLoad = Math.max(0, thermal - 70) * 0.04;
      const vibLoad = Math.max(0, vibration - 1.0) * 0.18;
      const baseLoad = 0.018;
      const rul = Math.max(0, prev.rul - (baseLoad + thermalLoad + vibLoad));

      const next: TelemetryFrame = {
        t,
        speed: Math.max(0, speed),
        altitude: Math.max(0, altitude),
        thermal,
        power: Math.max(0, power),
        vibration: Math.max(0, vibration),
        rul,
      };

      lastRef.current = next;
      pushFrame(next);
    }, intervalMs);

    return () => clearInterval(id);
  }, [missionRunning, intervalMs, pushFrame, current, resetMission, mode]);
}
