'use client';
import { useEffect, useState } from 'react';
import { fetchResults, type ResultRow } from '@/lib/api';
import ResultsTable from '@/components/Lab/ResultsTable';
import TrainingCurves from '@/components/Lab/TrainingCurves';
import PerAssetOverlay from '@/components/Lab/PerAssetOverlay';

export default function LabPage() {
  const [rows, setRows] = useState<ResultRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetchResults().then(setRows).catch((e) => setErr(e.message));
  }, []);

  return (
    <div className="h-full overflow-y-auto p-3 scrollbar-hud">
      <div className="mx-auto max-w-7xl space-y-3">
        <div className="holo-panel px-4 py-3">
          <div className="text-[11px] uppercase tracking-[0.4em] text-hud-cyan/80">SYSTEM // MODEL LAB</div>
          <h1 className="mt-1 text-2xl font-bold uppercase tracking-[0.2em] text-hud-cyan glitch-text">
            Predictive Maintenance · Architecture Comparison
          </h1>
          <p className="mt-2 max-w-3xl text-xs leading-relaxed text-hud-dim">
            LSTM, Transformer, and 1D-CNN trained on the NASA C-MAPSS turbofan benchmark
            (FD001–FD004) and on a synthesized multirotor UAV fleet (200 flights, 3 fault modes).
            Results below are loaded from <code className="text-hud-cyan">public/data/results.json</code>,
            written by <code className="text-hud-cyan">python -m ml.export</code>.
          </p>
        </div>

        {err && (
          <div className="holo-panel border-hud-red/60 p-3 text-xs text-hud-red">
            // FAILED TO LOAD RESULTS: {err}
          </div>
        )}

        {!rows && !err && (
          <div className="holo-panel p-3 text-xs text-hud-dim">// LOADING...</div>
        )}

        {rows && (
          <>
            <section className="holo-panel p-3">
              <div className="mb-2 text-[11px] uppercase tracking-[0.3em] text-hud-cyan">
                // TEST METRICS · ★ = BEST RMSE PER DATASET
              </div>
              <ResultsTable rows={rows} />
              <div className="mt-2 text-[10px] uppercase tracking-[0.25em] text-hud-dim">
                C-MAPSS = NASA turbofan engine degradation, RMSE on piecewise-linear RUL (clip 125).
                Score = PHM 2008 asymmetric scoring function (lower is better).
                UAV = synthesized multirotor flights, RMSE on normalized health-index RUL (clip 100).
              </div>
            </section>

            <section>
              <TrainingCurves rows={rows} />
            </section>

            <section>
              <PerAssetOverlay />
            </section>
          </>
        )}
      </div>
    </div>
  );
}
