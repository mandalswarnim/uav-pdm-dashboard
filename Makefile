.PHONY: setup fetch synth train export figures all clean backend

PY := python3

setup:
	$(PY) -m pip install -r requirements.txt

fetch:
	$(PY) -m scripts.fetch_cmapss

synth:
	$(PY) -m ml.data.uav_synth --n_drones 20 --flights_per_drone 10 --out data/uav_synth

train-cmapss:
	$(PY) -m ml.train --dataset cmapss --subset FD001 FD002 FD003 FD004 --arch lstm transformer cnn

train-uav:
	$(PY) -m ml.train --dataset uav --arch lstm transformer cnn

train: train-cmapss train-uav

export:
	$(PY) -m ml.export

figures:
	$(PY) -m ml.figures

phase1: fetch synth train export figures
	@echo "Phase 1 complete. Check artifacts/, public/data/, thesis/figures/, thesis/tables/."

backend:
	$(PY) -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload

clean:
	rm -rf artifacts/checkpoints artifacts/runs public/data thesis/figures thesis/tables
