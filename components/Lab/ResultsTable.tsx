'use client';
import type { ResultRow } from '@/lib/api';

const fmt = (v: number | null, d = 2) => (v == null ? '—' : v.toFixed(d));

export default function ResultsTable({ rows }: { rows: ResultRow[] }) {
  // Group: cmapss first, then uav.
  const sorted = [...rows].sort((a, b) =>
    (a.dataset === b.dataset ? archOrder(a.arch) - archOrder(b.arch) : a.dataset.localeCompare(b.dataset)),
  );

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-[11px] uppercase tracking-[0.18em]">
        <thead className="text-hud-cyan">
          <tr className="border-b border-hud-cyan/40">
            <Th>Dataset</Th><Th>Arch</Th><Th>RMSE</Th><Th>Score</Th>
            <Th>Fault Acc</Th><Th>Epochs</Th><Th>Train Size</Th><Th>Test Size</Th><Th>Time (s)</Th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const best = isBestForDataset(rows, r);
            return (
              <tr key={`${r.dataset}-${r.arch}`} className="border-b border-hud-cyan/10 hover:bg-hud-cyan/5">
                <Td>{r.dataset.toUpperCase()}</Td>
                <Td className={best ? 'text-hud-cyan font-bold' : ''}>
                  {r.arch.toUpperCase()}{best ? ' ★' : ''}
                </Td>
                <Td className={best ? 'text-hud-cyan font-bold' : ''}>{fmt(r.rmse, 3)}</Td>
                <Td>{r.score == null ? '—' : r.score.toFixed(1)}</Td>
                <Td>{r.fault_acc == null ? '—' : (r.fault_acc * 100).toFixed(1) + '%'}</Td>
                <Td>{r.epochs}</Td>
                <Td>{r.train_size.toLocaleString()}</Td>
                <Td>{r.test_size.toLocaleString()}</Td>
                <Td>{r.seconds.toFixed(1)}</Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function archOrder(a: string) { return ['lstm', 'transformer', 'cnn'].indexOf(a); }

function isBestForDataset(rows: ResultRow[], r: ResultRow) {
  const peers = rows.filter((x) => x.dataset === r.dataset);
  const min = Math.min(...peers.map((x) => x.rmse));
  return r.rmse === min;
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-2 py-1.5 text-left">{children}</th>;
}
function Td({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-2 py-1.5 ${className}`}>{children}</td>;
}
