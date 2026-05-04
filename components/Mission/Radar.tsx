'use client';
import { useEffect, useRef } from 'react';
import { useDash } from '@/lib/store';

interface Blip { x: number; y: number; vx: number; vy: number; hostile: boolean; id: number }

export default function Radar() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const blipsRef = useRef<Blip[]>([]);
  const angleRef = useRef(0);
  const trailRef = useRef<{ x: number; y: number }[]>([]);
  const { current, missionRunning } = useDash();

  // Seed blips
  useEffect(() => {
    let id = 0;
    blipsRef.current = Array.from({ length: 7 }).map(() => ({
      id: id++,
      x: (Math.random() - 0.5) * 1.6,
      y: (Math.random() - 0.5) * 1.6,
      vx: (Math.random() - 0.5) * 0.04,
      vy: (Math.random() - 0.5) * 0.04,
      hostile: Math.random() < 0.3,
    }));
  }, []);

  useEffect(() => {
    const cvs = canvasRef.current; if (!cvs) return;
    const ctx = cvs.getContext('2d')!;
    let raf = 0;
    const dpr = window.devicePixelRatio || 1;

    const resize = () => {
      const r = cvs.getBoundingClientRect();
      cvs.width = r.width * dpr;
      cvs.height = r.height * dpr;
    };
    resize();
    window.addEventListener('resize', resize);

    const render = () => {
      const W = cvs.width, H = cvs.height;
      const cx = W / 2, cy = H / 2;
      const R = Math.min(W, H) / 2 - 8 * dpr;

      ctx.clearRect(0, 0, W, H);

      // Background
      ctx.fillStyle = 'rgba(0,12,22,0.55)';
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.fill();

      // Range rings
      ctx.strokeStyle = 'rgba(0,229,255,0.35)';
      ctx.lineWidth = 1 * dpr;
      for (let i = 1; i <= 4; i++) {
        ctx.beginPath(); ctx.arc(cx, cy, (R / 4) * i, 0, Math.PI * 2); ctx.stroke();
      }
      // Crosshair
      ctx.beginPath();
      ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
      ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
      ctx.stroke();

      // Sweep
      angleRef.current += 0.025;
      const a = angleRef.current;
      const grad = ctx.createConicGradient(a, cx, cy);
      grad.addColorStop(0, 'rgba(0,229,255,0.0)');
      grad.addColorStop(0.06, 'rgba(0,229,255,0.45)');
      grad.addColorStop(0.12, 'rgba(0,229,255,0.0)');
      ctx.save();
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.clip();
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
      ctx.restore();

      // Update + draw blips
      const blips = blipsRef.current;
      blips.forEach((b) => {
        b.x += b.vx * 0.01; b.y += b.vy * 0.01;
        if (Math.hypot(b.x, b.y) > 0.95) { b.vx *= -1; b.vy *= -1; }
        const px = cx + b.x * R;
        const py = cy + b.y * R;
        ctx.fillStyle = b.hostile ? '#ff2a2a' : '#00ff9c';
        ctx.shadowColor = b.hostile ? '#ff2a2a' : '#00ff9c';
        ctx.shadowBlur = 10;
        ctx.beginPath(); ctx.arc(px, py, 3 * dpr, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
      });

      // Self position (center) + flight path trail derived from telemetry tick
      if (missionRunning && current) {
        const t = current.t;
        const ox = Math.sin(t * 0.15) * R * 0.35;
        const oy = Math.cos(t * 0.1) * R * 0.35;
        trailRef.current.push({ x: cx + ox, y: cy + oy });
        if (trailRef.current.length > 80) trailRef.current.shift();

        ctx.strokeStyle = 'rgba(0,229,255,0.6)';
        ctx.lineWidth = 1.5 * dpr;
        ctx.beginPath();
        trailRef.current.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
        ctx.stroke();

        const last = trailRef.current[trailRef.current.length - 1];
        ctx.fillStyle = '#00e5ff';
        ctx.shadowColor = '#00e5ff'; ctx.shadowBlur = 14;
        ctx.beginPath(); ctx.arc(last.x, last.y, 5 * dpr, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
      } else {
        trailRef.current = [];
      }

      raf = requestAnimationFrame(render);
    };
    render();
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
  }, [missionRunning, current]);

  return <canvas ref={canvasRef} className="h-full w-full" />;
}
