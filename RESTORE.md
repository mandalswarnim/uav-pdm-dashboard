# Restoring this project after cleanup (2026-06-11)

To free disk space, the regenerable parts of this project were deleted:
`.venv/` (Python env), `node_modules/`, `.next/` (build cache), and `data/`
(C-MAPSS + synthetic UAV parquet). All source code, `thesis/`, `Final Draft/`,
trained checkpoints (`artifacts/`), and baked dashboard JSON (`public/data/`)
were kept.

To work on this project again:

```bash
# Python env + deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Frontend deps
npm install

# Datasets (only needed for retraining)
make fetch   # downloads NASA C-MAPSS (S3 mirror, GitHub fallback)
make synth   # regenerates the synthetic UAV fleet (~2 s)
```

The trained model checkpoints in `artifacts/checkpoints/` were kept, so the
backend live-inference demo (`make backend`) and `make export` work without
retraining once deps are reinstalled.

Note: the portfolio website does NOT depend on this folder — the dashboard
was ported into the main repo (`app/uav/`, `components/uav/`, `lib/uav/`,
`public/uav/data/`) and this folder is gitignored.
