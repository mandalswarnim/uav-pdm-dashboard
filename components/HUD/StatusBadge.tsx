import type { Asset } from '@/lib/assets';

const styles: Record<Asset['status'], string> = {
  NOMINAL: 'border-hud-cyan/60 text-hud-cyan shadow-glow',
  WARNING: 'border-hud-amber/70 text-hud-amber shadow-glowAmber',
  CRITICAL: 'border-hud-red/80 text-hud-red shadow-glowRed animate-pulseRed',
};

export default function StatusBadge({ status }: { status: Asset['status'] }) {
  return (
    <span
      className={`inline-block border px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.25em] ${styles[status]}`}
    >
      {status}
    </span>
  );
}
