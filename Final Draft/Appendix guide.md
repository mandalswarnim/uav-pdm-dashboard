# Appendix Guide

Repository root: `/Volumes/Work/uav_predicition_dashboard`

Purpose: this is an extraction guide only. It tells you what to capture manually for the MSc dissertation appendix. It does not write appendix prose.

General screenshot rules:

- Use editor screenshots with filename tab and line numbers visible.
- Crop out the file tree, minimap, terminal prompt, unrelated imports, and unrelated code.
- For source code, prefer about 1000-1200 px wide screenshots so line numbers and identifiers remain readable.
- For dashboard screenshots, hide browser developer tools and avoid loading/error states.
- For figures, use the PDF versions in `thesis/figures/` when possible. Use PNG only if the document editor handles PNG better.

## Appendix A - Reproducibility

| Item | Exact file path | Exact class/function name | Approx. lines | What should be visible | Crop out | Recommended size | Recommended caption | Placement |
|---|---|---|---|---|---|---|---|---|
| Python dependencies | `requirements.txt` | N/A | 1-12 | Full dependency list, especially `torch`, `scikit-learn`, `captum`, `fastapi`, `uvicorn`, `pyarrow` | File tree, terminal | 900x420 | Python dependencies for ML training, explainability, and FastAPI inference. | Appendix A |
| Frontend dependencies | `package.json` | N/A | 5-30 | `scripts`, `dependencies`, and `devDependencies` | `package-lock.json`, editor sidebar | 900x600 | Next.js dashboard scripts and frontend runtime dependencies. | Appendix A |
| Environment setup targets | `Makefile` | Make targets: `setup`, `fetch`, `synth`, `train-cmapss`, `train-uav`, `export`, `figures`, `backend` | 5-35 | All reproducibility targets from setup through backend launch | `clean` target if space is tight | 1000x700 | Reproducibility commands for data generation, training, export, figures, and backend. | Appendix A |
| Central config and seed | `ml/config.py` | `SEED`, `CMAPSS`, `UAV` | 25-48 | `SEED = 1337`, sequence lengths, epochs, learning rate, UAV fault classes | Path constants above line 25 if space is tight | 1000x700 | Central hyperparameter and random seed configuration. | Appendix A |
| Training seed setup | `ml/train.py` | `_seed_everything` | 45-49 | Python, NumPy, PyTorch, and CUDA seed calls | Dataset adapters | 900x420 | Deterministic seed setup for training runs. | Appendix A |
| UAV generator seed | `ml/data/uav_synth.py` | `generate_fleet`, `main` | 148-150 and 280-287 | `seed: int = SEED`, `np.random.default_rng(seed)`, CLI `--seed` | Full simulator body | 1000x520 | Seeded procedural UAV fleet generation. | Appendix A |
| Device selection | `ml/config.py` | `get_device` | 51-56 | MPS, CUDA, then CPU fallback | Hyperparameters above | 900x420 | Device selection logic for Apple MPS, CUDA, and CPU execution. | Appendix A |
| Device usage in training | `ml/train.py` | `run` | 152-155 and 168-170 | `_seed_everything()`, `device = get_device()`, model `.to(device)`, optimizer | Full training loop | 1000x520 | Training driver using selected compute device. | Appendix A |
| Device usage in backend | `backend/inference.py` | `UAVPredictor.__init__`, `UAVPredictor._load` | 25-31 and 36-47 | Backend predictor loads model on selected device | Normalization code | 1000x600 | Live inference model loading on selected compute device. | Appendix A |

Commands to copy into Appendix A:

```bash
python3 -m pip install -r requirements.txt
npm install
make fetch
make synth
make train-cmapss
make train-uav
make export
make figures
make backend
npm run dev
```

## Appendix B - Code Listings

### B.1 LSTM

| Field | Instruction |
|---|---|
| Exact file path | `ml/models/lstm.py` |
| Exact class/function name | `LSTMRegressor`, `LSTMRegressor.__init__`, `LSTMRegressor.forward` |
| Approx. line numbers | 7-31 |
| What portion should be visible | Class name, `arch_name`, `nn.LSTM`, RUL head, optional fault head, `forward`, `h_last`, returned tuple |
| What should be cropped out | Imports and top docstring if space is tight |
| Recommended screenshot size | 1000x650 |
| Recommended caption | Stacked LSTM architecture with RUL regression and optional auxiliary fault-classification head. |
| Appendix section placement | Appendix B.1 |

### B.2 CNN

| Field | Instruction |
|---|---|
| Exact file path | `ml/models/cnn.py` |
| Exact class/function name | `CNNRegressor`, `CNNRegressor.__init__`, `CNNRegressor.forward` |
| Approx. line numbers | 11-44 |
| What portion should be visible | Class name, Conv1d block loop, BatchNorm/ReLU, global average pooling, RUL head, optional fault head |
| What should be cropped out | Top docstring and imports if space is tight |
| Recommended screenshot size | 1000x700 |
| Recommended caption | Temporal 1D-CNN baseline architecture for sequence-to-one RUL prediction. |
| Appendix section placement | Appendix B.2 |

### B.3 Transformer

This is the highest-priority appendix item. Capture these three screenshots separately.

| Item | Exact file path | Exact class/function name | Approx. lines | What should be visible | Crop out | Recommended size | Recommended caption | Placement |
|---|---|---|---|---|---|---|---|---|
| Positional encoding | `ml/models/transformer.py` | `_PositionalEncoding` | 11-22 | Sine/cosine positional encoding and registered buffer | Imports | 900x500 | Positional encoding used before Transformer encoder layers. | Appendix B.3 |
| Attention implementation | `ml/models/transformer.py` | `_AttnEncoderLayer` | 25-42 | `nn.MultiheadAttention`, `need_weights=True`, residual connection, layer norms, feed-forward block, `return x, attn_w` | Positional encoding above if capturing separately | 1100x650 | Transformer encoder layer exposing self-attention weights. | Appendix B.3 |
| Encoder block and model output | `ml/models/transformer.py` | `TransformerRegressor` | 45-77 | Projection, positional encoding, `ModuleList`, RUL head, fault head, mean pooling, return extras with `attn` | Imports and docstring | 1100x750 | Transformer RUL regressor returning final-layer attention for explainability. | Appendix B.3 |
| Attention extraction | `ml/xai.py` | `extract_attention` | 16-22 | `model.eval()`, model forward, fallback zero matrix, return `extras['attn']` | Integrated Gradients below if separate | 900x420 | Extraction of final-layer Transformer attention weights. | Appendix B.3 |
| Attention export logic | `ml/export.py` | `_bake_cmapss_assets`, `_bake_uav_assets` | 126-140 and 242-247 | Transformer-only attention extraction, `attn.tolist()`, IG nearby for context | Long JSON object fields | 1200x750 | Static export of Transformer attention matrices into dashboard artifacts. | Appendix B.3 |

### B.4 Multi-task Loss

| Item | Exact file path | Exact class/function name | Approx. lines | What should be visible | Crop out | Recommended size | Recommended caption | Placement |
|---|---|---|---|---|---|---|---|---|
| MSE and CrossEntropy | `ml/train.py` | `_train_one_epoch` | 52-70 | `nn.MSELoss()`, `nn.CrossEntropyLoss()`, `loss = mse(...)`, `loss + fault_loss_w * ce(...)`, backward and optimizer step | Evaluation function below | 1100x650 | Multi-task training loss combining RUL MSE with auxiliary fault classification. | Appendix B.4 |
| Lambda value | `ml/train.py` | `run` | 157-163 | C-MAPSS `fault_w = 0.0`, UAV `fault_w = 0.4` | Model saving code | 900x420 | UAV auxiliary fault-loss weighting with lambda set to 0.4. | Appendix B.4 |
| Scheduler | `ml/train.py` | `run` | 168-181 | AdamW, `CosineAnnealingLR`, `sched.step()`, history appending `lr` | Test evaluation below line 191 | 1100x650 | Cosine learning-rate scheduler and training-history recording. | Appendix B.4 |

### B.5 Explainability

| Item | Exact file path | Exact class/function name | Approx. lines | What should be visible | Crop out | Recommended size | Recommended caption | Placement |
|---|---|---|---|---|---|---|---|---|
| Captum dependency | `requirements.txt` | N/A | 7 | `captum>=0.7` | Other dependencies if showing as inset | 700x250 | Captum dependency used for Integrated Gradients. | Appendix B.5 |
| Integrated Gradients | `ml/xai.py` | `integrated_gradients` | 13 and 25-41 | Captum import, `_rul_only`, zero baseline, `ig.attribute`, returned attribution | `sensor_importance` if making separate shot | 1000x650 | Captum Integrated Gradients implementation for RUL-output attribution. | Appendix B.5 |
| Attention extraction | `ml/xai.py` | `extract_attention` | 16-22 | Pulls `extras['attn']` from model forward output | File docstring | 900x420 | Transformer attention extraction utility. | Appendix B.5 |
| Sensor importance | `ml/xai.py` | `sensor_importance` | 44-46 | Mean absolute attribution over time | Anything above line 44 if separate | 800x300 | Collapse of time-feature attributions into per-sensor importance. | Appendix B.5 |
| XAI export | `ml/export.py` | `_bake_cmapss_assets`, `_bake_uav_assets` | 126-140 and 242-247 | `extract_attention`, `integrated_gradients`, normalized feature importance | Per-asset metadata fields | 1200x700 | Export of attention and Integrated Gradients outputs into static asset JSON. | Appendix B.5 |

### B.6 UAV Degradation

| Item | Exact file path | Exact class/function name | Approx. lines | What should be visible | Crop out | Recommended size | Recommended caption | Placement |
|---|---|---|---|---|---|---|---|---|
| Degradation modes overview | `ml/data/uav_synth.py` | Module docstring | 1-23 | Three injected fault modes: bearing wear, ESC thermal, battery degradation | Imports | 1000x550 | Synthetic UAV degradation modes used to create learnable PdM signals. | Appendix B.6 |
| End-of-life constant | `ml/data/uav_synth.py` | `END_OF_LIFE_HOURS`, `FAULT_NAMES` | 45-50 | `DT`, comments, `END_OF_LIFE_HOURS = 4.0`, fault names | Phase table above unless useful | 900x420 | End-of-life horizon and fault-class labels for synthetic UAV data. | Appendix B.6 |
| Fault injection mechanics | `ml/data/uav_synth.py` | `_simulate_flight` | 70-115 | `life_frac`, bearing wear lines, ESC thermal lines, battery degradation voltage/internal resistance lines | DataFrame assembly from line 127 onward | 1200x800 | Bearing, ESC thermal, and battery degradation injected into synthetic flight telemetry. | Appendix B.6 |
| RUL label generation | `ml/data/uav_synth.py` | `generate_fleet` | 170-185 | RUL tied to `END_OF_LIFE_HOURS`, `rul_at_landing`, `nominal_eol` | Full file loop if too large | 1000x520 | RUL labels generated from cumulative flight hours and synthetic EOL horizon. | Appendix B.6 |

### B.7 Static Artifact Contract

| Item | Exact file path | Exact class/function name | Approx. lines | What should be visible | Crop out | Recommended size | Recommended caption | Placement |
|---|---|---|---|---|---|---|---|---|
| TypeScript contract | `lib/api.ts` | `AssetDetail`, `Manifest`, `ResultRow`, `fetchManifest`, `fetchAsset`, `fetchResults` | 6-55 | Interfaces for asset detail, manifest, results rows, and fetch functions | Top comments if space is tight | 1100x750 | Static artifact contract consumed by the Next.js dashboard. | Appendix B.7 |
| Asset export object | `ml/export.py` | `_bake_cmapss_assets`, `_bake_uav_assets` | 164-185 and 265-280 | Fields written into per-asset JSON including `attention_2d` and `sensor_importance` | Prediction loops above | 1200x800 | Export of per-asset static JSON artifacts. | Appendix B.7 |
| Results export | `ml/export.py` | `_bake_results_table` | 288-303 | Rows from `artifacts/runs/*.json`, metrics, history, write to `public/data/results.json` | CSV/TeX write if making separate shot | 1100x650 | Export of model metrics and training histories to `results.json`. | Appendix B.7 |
| Manifest export | `ml/export.py` | `main` | 328-340 | `manifest = {'assets': ..., 'generated_at': ...}` and write to `public/data/manifest.json` | Print statements below | 1000x520 | Static dashboard manifest generation. | Appendix B.7 |
| Manifest example | `public/data/manifest.json` | N/A | 1-10 and 100 | One asset object plus `generated_at` | Full asset list | 900x520 | Example dashboard manifest entry. | Appendix B.7 |
| Results example | `public/data/results.json` | N/A | 1-18, or UAV Transformer row around 1147-1158 | One result row with `dataset`, `arch`, metrics, and first history item | Full 1300-line history | 1000x650 | Example model-result artifact including metric fields and training history. | Appendix B.7 |

Recommended JSON fragment for dissertation:

```json
{
  "assets": [
    {
      "id": "AGM-09",
      "name": "HELLFIRE-IX",
      "class": "MISSILE-AGM",
      "rul": 118.17030334472656,
      "status": "NOMINAL",
      "data_source": "C-MAPSS"
    }
  ],
  "generated_at": "2026-06-04T00:22:30.706184+00:00"
}
```

```json
{
  "dataset": "uav",
  "arch": "transformer",
  "rmse": 3.447,
  "fault_acc": 0.955,
  "epochs": 30,
  "history": [
    {
      "epoch": 1,
      "train_loss": 322.7950866859604,
      "val_rmse": 6.2131476402282715,
      "lr": 0.0009972609476841367
    }
  ]
}
```

## Appendix C - Dashboard Screenshots

Before capturing:

1. Run `npm run dev`.
2. Open the dashboard in a browser.
3. Use a clean browser window with no dev tools.
4. Use the route paths below directly, or use the top navigation in `components/HUD/TopNav.tsx` lines 6-12.

| Page | Route | Component | How to navigate | Browser size | What should be visible | What should NOT be visible | Caption | Placement | Priority |
|---|---|---|---|---|---|---|---|---|---|
| Armory | `/armory` | `app/armory/page.tsx::ArmoryPage`; `components/Armory/AssetCarousel.tsx::AssetCarousel` | Click `ARMORY` in top nav | 1440x900 | Top nav, fleet roster, 3D carousel, status colors, selected asset dossier | Loading state, file tree, browser dev tools, unrelated home page | Armory view showing fleet roster, selected asset health, and 3D asset carousel. | Appendix C | High |
| Mission | `/mission` | `app/mission/page.tsx::MissionPage` | Click `MISSION`; use `PROCEDURAL`; click deploy; wait for charts to populate | 1440x900 | Radar, thermal/vibration/power charts, active asset, health bar, telemetry panel, mission controls | Empty charts, offline live backend message unless documenting live mode, dev tools | Mission view with procedural telemetry, radar, health, and control panels. | Appendix C | High |
| Diagnostics | `/diagnostics` | `app/diagnostics/page.tsx::DiagnosticsPage`; `components/Diagnostics/AttentionHeatmap.tsx::AttentionHeatmap`; `components/Diagnostics/MaintenanceReadout.tsx::MaintenanceReadout` | Click `DIAGNOSTICS`; select `CRZ-11` or another critical asset | 1440x900 | Subject list, digital twin wireframe, Transformer self-attention heatmap, forecast, prescribed action, sensor contributions | Loading dossier state, attention unavailable state, dev tools | Diagnostics view combining digital twin, Transformer attention, and Integrated Gradients sensor contributions. | Appendix C | Critical |
| Lab | `/lab` | `app/lab/page.tsx::LabPage`; `components/Lab/ResultsTable.tsx::ResultsTable`; `components/Lab/TrainingCurves.tsx::TrainingCurves` | Click `LAB` | 1440x1000 | Header, metrics table with all columns, C-MAPSS and UAV validation RMSE charts | Cut-off table columns, loading state, failed-to-load message | Model Lab comparison of LSTM, Transformer, and CNN across C-MAPSS and UAV-Synth. | Appendix C | High |

Source screenshots for Appendix C, if needed:

| Item | Exact file path | Exact class/function | Approx. lines | Visible region | Crop out | Recommended size | Caption | Placement |
|---|---|---|---|---|---|---|---|
| Navigation routes | `components/HUD/TopNav.tsx` | `TopNav`, `links` | 6-12 and 25-58 | Route labels `CMD`, `ARMORY`, `MISSION`, `DIAGNOSTICS`, `LAB` | Clock update internals if space is tight | 1000x600 | Dashboard navigation routes. | Appendix C intro |
| Armory page code | `app/armory/page.tsx` | `ArmoryPage` | 10-126 | Three-column layout: roster, carousel, dossier | Row helper if space is tight | 1200x800 | Armory page component structure. | Appendix C |
| Mission page code | `app/mission/page.tsx` | `MissionPage` | 15-126 | Radar/charts, simulation mode selector, telemetry/live controls, mission buttons | `ModeBtn` helper unless needed | 1200x850 | Mission page component structure. | Appendix C |
| Diagnostics page code | `app/diagnostics/page.tsx` | `DiagnosticsPage` | 11-99 | Subject list, wireframe, attention heatmap, maintenance readout | Loading guard if space is tight | 1200x800 | Diagnostics page component structure. | Appendix C |
| Lab page code | `app/lab/page.tsx` | `LabPage` | 8-68 | Fetch results, table, training curves, per-asset overlay | Imports | 1000x650 | Model Lab page component structure. | Appendix C |

## Appendix D - Results

| Item | Exact file path | Exact figure to export | Exact class/function or data item | Approx. lines | What to capture | Caption | Placement | Priority |
|---|---|---|---|---|---|---|---|---|
| Compact results table | `thesis/tables/results.csv` | Optional: `thesis/figures/fig_results_bars.pdf` | N/A | 1-7 | CSV rows for dataset, architecture, RMSE, score, fault accuracy, epochs, seconds | Test RMSE and fault-classification performance across architectures and datasets. | Appendix D | High |
| Results chart | `thesis/figures/fig_results_bars.pdf` or `.png` | `fig_results_bars.pdf` preferred | `ml/figures.py::fig_results_bars` | Source logic 134-152 | Export the figure, not a code screenshot, unless documenting generation code | Test RMSE comparison for C-MAPSS and UAV-Synth model runs. | Appendix D | High |
| Training histories | `artifacts/runs/*.json` | `thesis/figures/fig_loss_curves.pdf` preferred | `artifacts/runs/uav_transformer.json` as representative | Representative JSON lines 1-18; figure logic `ml/figures.py` lines 49-67 | First history object from one run, plus exported loss-curve figure | Validation RMSE histories for LSTM, Transformer, and CNN runs. | Appendix D | High |
| Attention outputs | `public/data/assets/CRZ-11.json` | `thesis/figures/fig_attention_CRZ-11.pdf` preferred | `attention_2d` | JSON lines 519-545; figure logic `ml/figures.py` lines 94-109 | First rows of `attention_2d`; do not show full matrix | Transformer final-layer self-attention heatmap for asset CRZ-11. | Appendix D | Critical |
| Sensor importance outputs | `public/data/assets/SW-101.json`; also `public/data/assets/CRZ-11.json` | `thesis/figures/fig_sensor_importance.pdf` preferred | `sensor_importance` | `SW-101` lines 3836-3859; `CRZ-11` lines 1481-1501; figure logic `ml/figures.py` lines 112-131 | Sensor-importance array and anomaly block; do not show full sensor window | Integrated Gradients sensor attribution scores from baked asset artifacts. | Appendix D | High |
| RUL overlay | `thesis/figures/fig_rul_overlay.pdf` or `.png` | `fig_rul_overlay.pdf` preferred | `ml/figures.py::fig_rul_overlay` | Source logic 70-91 | Export the figure showing predicted versus truth RUL by asset | Per-asset predicted RUL compared with ground-truth RUL. | Appendix D | Medium |
| UAV degradation figure | `thesis/figures/fig_uav_degradation.pdf` or `.png` | `fig_uav_degradation.pdf` preferred | Related source: `ml/data/uav_synth.py::_simulate_flight` | Source lines 70-115 | Export figure if the appendix needs a visual degradation summary | Synthetic UAV degradation signals across fault modes. | Appendix D or B.6 supporting figure | Medium |

Available generated figures:

| Figure file | Native size or format | Recommended use |
|---|---|---|
| `thesis/figures/fig_architecture.pdf` | PDF, 1 page | Architecture overview if needed in main dissertation or appendix. |
| `thesis/figures/fig_attention_CRZ-11.pdf` | PDF, 1 page | Highest-priority attention output for Appendix D. |
| `thesis/figures/fig_loss_curves.pdf` | PDF, 1 page | Training-history evidence for Appendix D. |
| `thesis/figures/fig_results_bars.pdf` | PDF, 1 page | Results comparison for Appendix D. |
| `thesis/figures/fig_rul_overlay.pdf` | PDF, 1 page | Per-asset prediction evidence for Appendix D. |
| `thesis/figures/fig_sensor_importance.pdf` | PDF, 1 page | Sensor-importance evidence for Appendix D. |
| `thesis/figures/fig_uav_degradation.pdf` | PDF | UAV degradation visual support. |

## Final Checklist

| APPENDIX ITEM | FILE | WHAT TO CAPTURE | CAPTION | PRIORITY |
|---|---|---|---|---|
| A. Python dependencies | `requirements.txt` | Lines 1-12 | Python dependencies for ML training, explainability, and FastAPI inference. | High |
| A. Frontend dependencies | `package.json` | Lines 5-30 | Next.js dashboard scripts and frontend runtime dependencies. | High |
| A. Reproduction commands | `Makefile` | Lines 5-35 plus command block above | Reproducibility commands for data generation, training, export, figures, and backend. | High |
| A. Seed configuration | `ml/config.py`; `ml/train.py`; `ml/data/uav_synth.py` | `SEED`, `_seed_everything`, UAV generator `--seed` | Random seed configuration used for training and synthetic UAV generation. | High |
| A. Device selection | `ml/config.py`; `ml/train.py`; `backend/inference.py` | `get_device()` and usage in training/backend | Device selection logic for MPS, CUDA, and CPU. | High |
| B.1 LSTM | `ml/models/lstm.py` | `LSTMRegressor`, lines 7-31 | Stacked LSTM architecture with RUL and optional fault head. | High |
| B.2 CNN | `ml/models/cnn.py` | `CNNRegressor`, lines 11-44 | Temporal 1D-CNN baseline architecture. | High |
| B.3 Transformer attention | `ml/models/transformer.py` | `_AttnEncoderLayer`, lines 25-42 | Transformer encoder layer exposing self-attention weights. | Critical |
| B.3 Transformer encoder | `ml/models/transformer.py` | `TransformerRegressor`, lines 45-77 | Transformer RUL regressor returning final-layer attention. | Critical |
| B.3 Attention export | `ml/xai.py`; `ml/export.py` | `extract_attention`, export lines 126-140 and 242-247 | Attention extraction and static artifact export. | Critical |
| B.4 Multi-task loss | `ml/train.py` | `_train_one_epoch`, lines 52-70 | Multi-task MSE plus CrossEntropy loss. | High |
| B.4 Lambda and scheduler | `ml/train.py` | `fault_w = 0.4`, `CosineAnnealingLR`, lines 157-181 | UAV loss weighting and scheduler. | High |
| B.5 Explainability | `requirements.txt`; `ml/xai.py`; `ml/export.py` | Captum, IG, attention, sensor importance | Integrated Gradients and attention explainability implementation. | High |
| B.6 UAV degradation | `ml/data/uav_synth.py` | `END_OF_LIFE_HOURS`, `_simulate_flight`, RUL labels | Synthetic bearing, ESC thermal, and battery degradation model. | High |
| B.7 Static artifact contract | `lib/api.ts`; `ml/export.py`; `public/data/*.json` | TypeScript interfaces, export logic, compact JSON examples | Static dashboard artifact contract and JSON examples. | High |
| C. Armory dashboard | `/armory`; `app/armory/page.tsx` | Browser screenshot at 1440x900 | Armory view showing fleet roster, selected asset health, and 3D asset carousel. | High |
| C. Mission dashboard | `/mission`; `app/mission/page.tsx` | Browser screenshot at 1440x900 after deploy | Mission view with procedural telemetry, radar, health, and controls. | High |
| C. Diagnostics dashboard | `/diagnostics`; `app/diagnostics/page.tsx` | Browser screenshot at 1440x900, select critical asset | Diagnostics view with digital twin, attention heatmap, and IG contributions. | Critical |
| C. Lab dashboard | `/lab`; `app/lab/page.tsx` | Browser screenshot at 1440x1000 | Model Lab comparison table and validation RMSE charts. | High |
| D. Results table | `thesis/tables/results.csv` | Lines 1-7 | Test RMSE and fault-classification results. | High |
| D. Results bars | `thesis/figures/fig_results_bars.pdf` | Export figure | Test RMSE comparison for C-MAPSS and UAV-Synth. | High |
| D. Training histories | `artifacts/runs/*.json`; `thesis/figures/fig_loss_curves.pdf` | One representative JSON history plus exported figure | Validation RMSE histories for all model architectures. | High |
| D. Attention output | `public/data/assets/CRZ-11.json`; `thesis/figures/fig_attention_CRZ-11.pdf` | `attention_2d` start plus exported figure | Transformer final-layer self-attention heatmap for CRZ-11. | Critical |
| D. Sensor importance | `public/data/assets/SW-101.json`; `thesis/figures/fig_sensor_importance.pdf` | `sensor_importance` plus exported figure | Integrated Gradients sensor attribution scores. | High |
