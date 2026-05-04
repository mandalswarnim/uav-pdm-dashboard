'use client';
import { useEffect, useMemo, useRef } from 'react';

interface Props {
  /** (T, T) attention weights from the Transformer's last encoder layer */
  matrix: number[][] | null;
  height?: number;
}

/**
 * Renders a (T × T) self-attention heatmap onto a canvas with a viridis-ish ramp.
 * Y axis = query timestep (later in sequence = top), X axis = attended timestep.
 * Saves us from importing a chart library for a single image-style plot.
 */
export default function AttentionHeatmap({ matrix, height = 220 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const stats = useMemo(() => {
    if (!matrix || !matrix.length) return null;
    let lo = Infinity, hi = -Infinity;
    for (const row of matrix) for (const v of row) {
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
    return { lo, hi };
  }, [matrix]);

  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs || !matrix || !stats) return;
    const T = matrix.length;
    const dpr = window.devicePixelRatio || 1;
    const rect = cvs.getBoundingClientRect();
    cvs.width = rect.width * dpr;
    cvs.height = rect.height * dpr;
    const ctx = cvs.getContext('2d')!;
    const cellW = cvs.width / T;
    const cellH = cvs.height / T;
    const range = Math.max(1e-6, stats.hi - stats.lo);

    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        const v = (matrix[i][j] - stats.lo) / range;
        ctx.fillStyle = ramp(v);
        // y is flipped so query-step 0 is bottom, latest is top
        ctx.fillRect(j * cellW, (T - 1 - i) * cellH, cellW + 1, cellH + 1);
      }
    }
  }, [matrix, stats]);

  if (!matrix) {
    return (
      <div
        className="flex items-center justify-center border border-hud-cyan/30 text-[10px] uppercase tracking-[0.25em] text-hud-dim"
        style={{ height }}
      >
        // ATTENTION UNAVAILABLE — model lacks self-attention
      </div>
    );
  }

  return (
    <div className="relative" style={{ height }}>
      <canvas ref={canvasRef} className="h-full w-full" />
      <div className="pointer-events-none absolute bottom-1 left-1 text-[9px] uppercase tracking-[0.2em] text-hud-cyan/80">
        Q-step 0 →
      </div>
      <div className="pointer-events-none absolute right-1 top-1 text-[9px] uppercase tracking-[0.2em] text-hud-cyan/80">
        K-step (attended)
      </div>
    </div>
  );
}

/** Approximate viridis: deep navy → cyan → yellow */
function ramp(v: number): string {
  const x = Math.max(0, Math.min(1, v));
  const r = Math.round(255 * Math.pow(x, 2.2));
  const g = Math.round(255 * Math.pow(x, 0.85));
  const b = Math.round(255 * (0.4 + 0.6 * (1 - x)));
  return `rgb(${r},${g},${b})`;
}
