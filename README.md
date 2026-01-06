# ECG Edge Robustness

This repository implements a full edge-feasible ECG arrhythmia classification pipeline following the Codex specification. It includes data preprocessing for MIT-BIH and INCART, compact and reference neural network models, augmentation-aware training, evaluation utilities, and tooling to check datasets and deployment suitability.

## Quick start

1. Ensure the PhysioNet MIT-BIH and INCART datasets are placed under `data/mitbih` and `data/incart` respectively.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the full pipeline: `bash scripts/run_all.sh`.

See `IMPLEMENTATION.md` for detailed design goals.
