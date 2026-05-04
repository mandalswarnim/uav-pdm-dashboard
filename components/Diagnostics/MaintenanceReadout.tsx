'use client';
import type { AssetDetail } from '@/lib/api';

interface Props { detail: AssetDetail | null }

export default function MaintenanceReadout({ detail }: Props) {
  if (!detail) {
    return <div className="text-xs text-hud-dim">// Loading dossier from baked artifacts...</div>;
  }

  const sev = detail.anomaly?.severity ?? 0;
  const action = !detail.anomaly
    ? 'No action required. Continue scheduled inspections.'
    : sev > 0.8
      ? 'GROUND IMMEDIATELY. Replace affected component before next sortie.'
      : sev > 0.5
        ? 'Schedule depot-level maintenance within 48 hours.'
        : 'Monitor next 3 sorties; recheck telemetry trend.';

  // Top-N most-influential sensors from per-feature IG attribution.
  const importance = detail.sensor_importance ?? [];
  const ranked = importance
    .map((v, i) => ({ name: detail.sensor_names[i] ?? `f${i}`, v }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 8);

  return (
    <div className="space-y-3">
      <Section title="// AI CLASSIFICATION">
        <p className="text-xs leading-relaxed text-hud-dim">
          {detail.data_source === 'C-MAPSS'
            ? 'Transformer + LSTM ensemble attributes RUL drop primarily to '
            : 'UAV PdM model classifies fault as '}
          <span className="text-hud-red">{detail.anomaly?.component ?? 'no anomalous channel'}</span>.
          {detail.anomaly?.predicted_fault && (
            <> Predicted fault class: <span className="text-hud-amber">{detail.anomaly.predicted_fault}</span>
              {detail.anomaly.true_fault && <> (truth: {detail.anomaly.true_fault})</>}.
            </>
          )}
        </p>
      </Section>

      <Section title="// FORECAST">
        <Row k="Pred. RUL"   v={`${detail.rul.toFixed(1)} u`} alert={detail.rul < 35} />
        <Row k="Truth RUL"   v={detail.rul_truth != null ? `${detail.rul_truth.toFixed(1)} u` : '—'} />
        <Row k="Best model"  v={detail.best_arch?.toUpperCase() ?? '—'} />
        {detail.predictions_per_arch && (
          <>
            <Row k="LSTM pred"        v={fmtMaybe(detail.predictions_per_arch.lstm)} />
            <Row k="Transformer pred" v={fmtMaybe(detail.predictions_per_arch.transformer)} />
            <Row k="CNN pred"         v={fmtMaybe(detail.predictions_per_arch.cnn)} />
          </>
        )}
        <Row k="Failure mode" v={detail.anomaly?.note ?? '—'} />
      </Section>

      <Section title="// PRESCRIBED ACTION">
        <p className={`text-xs leading-relaxed ${sev > 0.8 ? 'text-hud-red' : sev > 0.5 ? 'text-hud-amber' : 'text-hud-cyan'}`}>
          {action}
        </p>
      </Section>

      <Section title="// SENSOR CONTRIBUTIONS · INTEGRATED GRADIENTS">
        {ranked.length === 0 ? (
          <div className="text-[10px] text-hud-dim">No attribution available.</div>
        ) : (
          ranked.map((s, i) => (
            <Bar
              key={s.name}
              label={s.name}
              v={s.v}
              color={i === 0 ? '#ff2a2a' : i === 1 ? '#ffb000' : '#00e5ff'}
            />
          ))
        )}
      </Section>
    </div>
  );
}

const fmtMaybe = (v: number | undefined) => (v == null ? '—' : v.toFixed(2));

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border-l-2 border-hud-cyan/50 pl-3">
      <div className="mb-1 text-[10px] uppercase tracking-[0.3em] text-hud-cyan/80">{title}</div>
      {children}
    </div>
  );
}

function Row({ k, v, alert }: { k: string; v: string; alert?: boolean }) {
  return (
    <div className={`flex justify-between border-b border-hud-cyan/10 py-0.5 text-[11px] uppercase tracking-[0.2em] ${alert ? 'text-hud-red' : ''}`}>
      <span className="text-hud-dim">{k}</span>
      <span className="font-bold">{v}</span>
    </div>
  );
}

function Bar({ label, v, color }: { label: string; v: number; color: string }) {
  return (
    <div className="my-1.5">
      <div className="flex justify-between text-[10px] uppercase tracking-[0.2em] text-hud-dim">
        <span>{label}</span>
        <span style={{ color }}>{(v * 100).toFixed(0)}%</span>
      </div>
      <div className="h-1.5 w-full bg-hud-cyan/10">
        <div className="h-full" style={{ width: `${v * 100}%`, background: color, boxShadow: `0 0 6px ${color}` }} />
      </div>
    </div>
  );
}
