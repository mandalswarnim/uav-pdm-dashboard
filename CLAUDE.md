# ARES PdM — Tactical UAV Predictive Maintenance Digital Twin

Master's-level Predictive Maintenance / CBM+ project: PyTorch models (LSTM,
Transformer, 1D-CNN) trained on NASA C-MAPSS + a synthesized multirotor UAV
fleet, served into a Next.js "Jarvis" HUD with live WebSocket inference.

## Layout

```
/                             repo root
├── app/                      Next.js 14 app router
│   ├── armory/               Fleet inventory · 3D carousel
│   ├── mission/              Live HUD · radar + telemetry + procedural OR live-inference mode
│   ├── diagnostics/          Digital twin · wireframe + attention heatmap + IG sensor importance
│   ├── lab/                  Model Lab · results table + training curves + per-asset prediction overlay
│   └── page.tsx              Landing
├── components/               React components grouped by view
│   ├── 3D/                   procedural Three.js geometry (UAVs, missiles)
│   ├── HUD/                  TopNav, StatusBadge
│   ├── Armory/               AssetCarousel
│   ├── Mission/              Radar, TelemetryPanel, HealthBar, LiveChart, LiveControls
│   ├── Diagnostics/          WireframeView, AttentionHeatmap, MaintenanceReadout
│   └── Lab/                  ResultsTable, TrainingCurves, PerAssetOverlay
├── lib/                      Frontend state + API/WS clients
│   ├── api.ts                static fetchers for /data/manifest.json, /data/assets/*, /data/results.json
│   ├── live.ts               WebSocket client for the FastAPI backend
│   ├── useLiveStream.ts      React hook bridging WS → store
│   ├── store.ts              Zustand store (assets, details cache, mission mode, live ticks)
│   ├── telemetry.ts          procedural client-side flight envelope (PROCEDURAL mode)
│   └── assets.ts             Asset / AssetClass / status types
│
├── ml/                       PyTorch ML pipeline
│   ├── config.py             paths, hyperparams, get_device() (MPS-aware)
│   ├── data/cmapss.py        load + preprocess FD001-FD004
│   ├── data/uav_synth.py     procedural multirotor generator + loader
│   ├── models/{lstm,transformer,cnn}.py
│   ├── train.py              unified driver (CLI: --dataset cmapss|uav --arch ...)
│   ├── xai.py                attention extraction + Integrated Gradients
│   ├── export.py             bakes per-asset JSON + results table for the dashboard
│   └── figures.py            publication matplotlib figures (PNG + PDF)
├── backend/                  FastAPI live inference service
│   ├── inference.py          loads UAV transformer; thread-safe predict()
│   ├── synth_stream.py       server-side flight generator + rolling window buffer
│   └── main.py               /healthz, POST /predict, WS /stream
├── scripts/fetch_cmapss.py   downloads NASA C-MAPSS (S3 + GitHub fallback)
│
├── public/data/              ← baked artifacts the dashboard reads (git-ignored)
│   ├── manifest.json
│   ├── assets/<id>.json      per-asset RUL + sensor window + attention + IG
│   └── results.json          Model Lab metrics
├── thesis/
│   ├── methodology.md        scaffold for thesis paste-in
│   ├── figures/              PNG + PDF (auto-generated)
│   └── tables/               results.csv + results.tex
├── data/                     C-MAPSS + UAV synth data (git-ignored)
└── artifacts/                checkpoints + per-run history JSON (git-ignored)
```

## Tech stack

- **Frontend** Next.js 14 app router, React 18, TypeScript, Tailwind, @react-three/fiber + drei, recharts, zustand
- **ML** PyTorch on Apple Silicon MPS, scikit-learn, pandas, matplotlib, captum
- **Backend** FastAPI + uvicorn, websockets, pydantic
- **Python**: 3.13.3 in `.venv/`

## How to run

Two long-running processes. Both are needed for the live demo; the dashboard
also works in PROCEDURAL mode without the backend.

```bash
# 1) Frontend dev server (auto-picks port if 3000 taken — currently 3001 here)
npm run dev

# 2) Backend (live UAV inference) on port 8001
make backend
# or:  .venv/bin/python -m uvicorn backend.main:app --port 8001 --reload
```

Then open http://localhost:3001 → CMD / ARMORY / MISSION / DIAGNOSTICS / LAB.

### Re-training / re-baking

```bash
make phase1     # full pipeline: fetch + synth + train all 6 + export + figures
make fetch      # C-MAPSS download (idempotent)
make synth      # regenerate UAV synth fleet (~2 s for 20 drones × 10 flights)
make train      # train all 3 archs on C-MAPSS + UAV (~35 min on MPS)
make train-uav  # UAV only (~10 min)
make export     # rebuild public/data/ + thesis/tables/ from current checkpoints
make figures    # rebuild thesis/figures/
```

After re-training, restart the backend so it picks up the new checkpoints:
`pkill -f "uvicorn backend.main" && make backend`.

## Dashboard ↔ ML contract

`ml/export.py` writes everything the frontend consumes:

- **`public/data/manifest.json`** — `{ assets: [{ id, name, class, rul, status, data_source }] }`
  hydrated into the Zustand store on app boot via `components/AppBootstrap.tsx`.
- **`public/data/assets/<id>.json`** (loaded on selection) — full `AssetDetail`:
  `predictions_per_arch`, `rmse_per_arch`, `sensor_names`, `sensor_window` (T×F),
  `attention_2d` (T×T), `sensor_importance` (F,), `anomaly` { component, severity, ... }.
  Drives Diagnostics wireframe + heatmap + sensor importance + per-asset bars.
- **`public/data/results.json`** — array of training-run objects with `history[]`;
  drives the Model Lab table + training curves.

If you change asset/result schemas in `ml/export.py`, also update `lib/api.ts`
(types) and the consumers (`MaintenanceReadout`, `PerAssetOverlay`, `ResultsTable`).

## Live inference protocol

WebSocket: `ws://127.0.0.1:8001/stream?fault=...&hours=...&rate_hz=...&stride=...`

Server emits, in order: one `meta` message, then `tick` messages (one per simulated tick
at 10 Hz default; `prediction` field present every `stride`-th tick once the rolling
window is full), then `end`. Client uses `lib/live.ts` (`LiveStream`) and
`lib/useLiveStream.ts` (auto-subscribes when `mode === 'LIVE'` && `missionRunning`).
`store.ts → ingestLiveTick` maps raw UAV channels onto the abstract `TelemetryFrame`
slots so all existing HUD components work unchanged.

## Key constants & gotchas

- **`END_OF_LIFE_HOURS = 4.0`** in `ml/data/uav_synth.py` — must stay small relative
  to total flight hours per drone (~3 h with 10 flights). If raised back to 200,
  RUL collapses to a near-constant 99 and models trivially predict it. Was the
  cause of the Phase-1.5-looked-fine, Phase-3-broke-everything bug.
- **CMAPSS RUL clip = 125** (cycles, not %) — the dashboard clamps display width to
  `min(100, rul)` for the roster bar. Don't conflate "RUL units" with "% of life".
- **`rul_to_status`** thresholds (`<35` CRITICAL, `<65` WARNING) live in two places
  on purpose: `ml/export.py` (server-side, applied to RUL units) and
  `lib/assets.ts` (client-side, when re-classifying live RUL). Keep them in sync.
- **MPS OOM** — running large val sets through Transformer in one batch will fail.
  Always batch via `_batched_predict(model, X, batch=256)` in `ml/export.py`.
- **C-MAPSS download** — NASA's PCoE link rots; primary mirror is S3
  (`phm-datasets.s3.amazonaws.com`), fallback is the hankroark GitHub raw mirror.
  See `scripts/fetch_cmapss.py`.
- **Ports** — the user's machine has a stale uvicorn on **8000** and frequently
  has 3000 taken. Default to **3001** for Next dev and **8001** for FastAPI.

## What's intentionally NOT here

- No GLTF assets — all 3D geometry is procedural in `components/3D/UAVModel.tsx`.
- Mission mode is split: PROCEDURAL (heuristic envelope, runs offline) vs.
  LIVE INFERENCE (FastAPI + trained model). Both are kept; toggling one off
  doesn't replace the other.
- C-MAPSS engines are baked-only (static replay). The live WebSocket is UAV-only.
- Fleet is curated to 12 assets (8 turbofan + 4 quadrotor) — not the full
  ~700-engine roster — by design, for the 3D carousel + demo polish.

## Reproducing results

End-to-end (Apple Silicon, ~35 min):
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make phase1
npm install && npm run dev   # in another shell
make backend                  # in a third shell
```

Headline metrics with default seeds (SEED=1337):
- C-MAPSS FD001-FD004 combined: Transformer RMSE 16.40 / Score 4074
- UAV synth (4 fault classes, 200 flights): Transformer RMSE 3.45 / Fault Acc 95.5 %
