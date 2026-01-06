# ECG Edge Robustness Project — Codex Implementation Spec (Drop-in Document)

**Purpose of this document:** Place this file at the root of your repository so that Codex (or any code-generation agent) can implement the project end-to-end, with checks that confirm the pipeline works.

---

## 0) What we are building (Version 1 — No KD initially)

We will implement an **edge-feasible single‑lead ECG arrhythmia classifier** and evaluate **cross‑dataset generalization** (domain shift):

- **Source domain:** MIT‑BIH Arrhythmia Database (single lead; MLII if available).
- **Target domain:** INCART 12‑lead Arrhythmia Database (use **only Lead II**).
- **Task:** AAMI‑style **5‑class** beat classification (N, S, V, F, Q).
- **Modeling:**
  - Teacher: ResNet1D (offline reference model; not required to deploy).
  - Student: Tiny depthwise‑separable 1D CNN (edge candidate; deployable).
- **Domain robustness method (Version 1):**
  - Train student normally (ERM baseline).
  - Train student with **ECG-realistic augmentations** (domain generalization).
  - Optional: AdaBN (BN-stat adaptation using unlabeled INCART-adapt).

**Important:** The main “edge” artifact is the **student**, not the teacher. The teacher exists to (1) validate pipeline and (2) establish an upper bound and later enable KD if needed.

---

## 1) Repository structure (Codex must create exactly this)

```
ecg-edge-robustness/
  CODEX_IMPLEMENTATION.md        # this file
  README.md
  requirements.txt
  .gitignore

  data/
    mitbih/                      # raw MIT-BIH WFDB files (.dat/.hea/.atr)
    incart/                      # raw INCART WFDB files (.dat/.hea/.atr)
  processed/
    mitbih/
      train.npz
      val.npz
      test.npz
    incart/
      adapt.npz                  # unlabeled for adaptation (AdaBN) — never used as test
      test.npz                   # held-out evaluation
  outputs/
    teacher/
      best.pt
      metrics.json
    student/
      best.pt
      metrics.json
    student_aug/
      best.pt
      metrics.json

  src/
    __init__.py

    configs/
      __init__.py
      splits.py                  # DS1/DS2 + exclusions

    data/
      __init__.py
      aami.py                    # mapping + ignore symbols
      wfdb_io.py                 # WFDB reading utils (lead selection, resampling)
      prepare_mitbih.py          # creates processed/mitbih/*.npz
      prepare_incart.py          # creates processed/incart/*.npz

    datasets/
      __init__.py
      npz_dataset.py             # loads .npz and returns (x,y)
      paired_loaders.py          # optional (source+target loaders if needed later)

    augment/
      __init__.py
      ecg_aug.py                 # PyTorch augmentations (domain generalization)

    models/
      __init__.py
      resnet1d.py                # teacher
      tinydscnn1d.py             # student

    train/
      __init__.py
      train_erm.py               # common supervised trainer
      train_utils.py             # seed, metrics, checkpointing, class imbalance sampler

    eval/
      __init__.py
      metrics.py                 # macro-F1, per-class F1, confusion, etc.
      evaluate.py                # evaluate a saved model on an .npz split

    tools/
      __init__.py
      env_check.py               # checks MPS/CPU, dependency versions
      dataset_check.py           # verifies expected WFDB files exist, counts records
      edge_report.py             # prints params, approx MACs, INT8 size estimate (weights only)

  scripts/
    run_all.sh                   # one-command pipeline (preprocess → train → eval)
```

---

## 2) Datasets: Do you need to download them and place folders locally?

**Yes.** Codex cannot reliably fetch PhysioNet datasets without your credentials/network context.

### Expected dataset folder contents

#### `data/mitbih/`
Must contain WFDB record triplets for required records, e.g.:
- `100.dat`, `100.hea`, `100.atr`
- ...
If your MIT‑BIH download uses a different annotation extension, adapt `prepare_mitbih.py`.

#### `data/incart/`
Must contain WFDB record triplets, often named like:
- `I01.dat`, `I01.hea`, `I01.atr`
- ...
If your INCART uses different annotation extensions, `prepare_incart.py` must match.

### Acceptance checks
`python -m src.tools.dataset_check --data_dir data/mitbih --dataset mitbih`
- must confirm the DS1/DS2 records exist.
`python -m src.tools.dataset_check --data_dir data/incart --dataset incart`
- must list record IDs found and confirm annotation files.

---

## 3) Environment setup (M1 Mac friendly)

### Create and activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Install dependencies
Codex must produce `requirements.txt`:

Minimal:
- numpy
- scipy
- wfdb
- scikit-learn
- tqdm
- matplotlib
- torch

Install:
```bash
pip install -r requirements.txt
```

### Verify environment
```bash
python -m src.tools.env_check
```

Pass criteria:
- prints torch version
- prints whether `mps` is available (on Apple Silicon) OR falls back to CPU gracefully

---

## 4) Data protocol (must be consistent across datasets)

### Canonical settings (for Version 1)
Codex should implement these as default CLI args:
- Canonical sampling rate: **250 Hz**
- Window type: **beat-centered**
- Window length at 250 Hz: **256 samples**
- Pre-R fraction: **0.30** (≈77 samples pre, 179 post)

### Preprocessing steps (applies to both datasets)
1. Read WFDB signal.
2. Select target lead:
   - MIT‑BIH: prefer `"MLII"` if present else channel 0.
   - INCART: prefer `"II"` if present else channel mapping or fallback.
3. Optional bandpass filter: 0.5–40 Hz (Butterworth 3rd order).
4. Resample to 250 Hz:
   - MIT‑BIH: 360→250
   - INCART: 257→250
5. Extract beat-centered windows using annotation sample indices (scaled if resampled).
6. Normalize each window:
   - default: **per-window z-score**
7. Map annotation symbols to AAMI 5 classes.
8. Save `.npz` with:
   - `X`: float32 array `[N, 1, L]`
   - `y`: int64 array `[N]`

---

## 5) AAMI 5-class mapping (must be explicit)

Codex must implement `src/data/aami.py` with:
- `AAMI_MAP` mapping symbols → class id
- `IGNORE_SYMBOLS` to skip non-beat markers

Default mapping:
- N: N, L, R, e, j  → 0
- S: A, a, J, S      → 1
- V: V, E            → 2
- F: F               → 3
- Q: Q, f, /, U      → 4

Exclusions:
- Paced MIT‑BIH records: {102,104,107,217}

---

## 6) Splits (no leakage rules)

### MIT‑BIH inter-patient split
- DS1 used for training/validation.
- DS2 held out for test.
- Within DS1, split by **record** into train/val (e.g., 80/20).

### INCART split
Split INCART records by **record**, not by beat:
- `incart/adapt`: unlabeled set used for AdaBN only
- `incart/test`: held-out evaluation

Default:
- adapt: first 60% of record IDs (sorted)
- test: remaining 40%

---

## 7) Models (teacher + student)

### Teacher: `ResNet1DTeacher`
- Stem: Conv7 s2 + BN + ReLU + MaxPool s2
- Stages: (32×2 blocks), (64×2 blocks downsample), (128×2 blocks downsample)
- Head: GAP → FC 128→64 → Dropout → FC 64→5

### Student: `TinyDSCNN1D`
- Stem: Conv7 s2 (1→16) + BN + ReLU
- DS blocks: depthwise conv3 + pointwise conv1:
  - 16→24 (downsample)
  - 24→32 (downsample)
  - 32→48 (no downsample)
  - 48→64 (downsample)
- Head: GAP → FC 64→5

---

## 8) Training (ERM baseline, then augmentation)

### Class imbalance handling (mandatory)
- WeightedRandomSampler OR weighted cross-entropy
- Default: WeightedRandomSampler based on inverse class frequency

### Trainer design
`src/train/train_erm.py` must:
- train any model with CE loss
- log train loss and val macro-F1 each epoch
- checkpoint best model by val macro-F1
- early stopping optional
- run on MPS if available else CPU

---

## 9) ECG augmentations (domain generalization)

`src/augment/ecg_aug.py` must implement PyTorch augmentations:
- baseline wander (low-frequency sine)
- powerline hum (50 Hz)
- gaussian noise
- amplitude scaling
- small time shift

Aug policy:
- mild + probabilistic
- optional ramp-up:
  - epochs 1–5: none
  - epochs 6–10: half-strength
  - epochs 11+: full

Experiments:
- Student-CE baseline (no aug)
- Student-CE+Aug (with aug)

---

## 10) Evaluation and reporting

`src/eval/evaluate.py` must compute:
- macro-F1
- per-class F1
- confusion matrix
- per-class precision/recall

Required eval runs:
1. Teacher on MIT‑BIH DS2
2. Student baseline on MIT‑BIH DS2 and INCART test
3. Student augmented on MIT‑BIH DS2 and INCART test

Save metrics to JSON under `outputs/*/metrics.json`.

---

## 11) Edge constraint reporting (bridge to “edge”)

`src/tools/edge_report.py` must print for a model:
- #parameters
- approximate weight size in KB (FP32 and INT8)
- rough MACs estimate for 1D conv layers

---

## 12) “Ensure it’s working” — required smoke tests (must pass)

### (A) Dataset presence check
```bash
python -m src.tools.dataset_check --dataset mitbih --data_dir data/mitbih
python -m src.tools.dataset_check --dataset incart --data_dir data/incart
```

### (B) Preprocess
```bash
python -m src.data.prepare_mitbih --data_dir data/mitbih --out_dir processed/mitbih --target_fs 250 --window_len 256
python -m src.data.prepare_incart --data_dir data/incart --out_dir processed/incart --target_fs 250 --window_len 256 --adapt_ratio 0.6
```

### (C) Train 1-epoch debug runs
```bash
python -m src.train.train_erm --model teacher --train_npz processed/mitbih/train.npz --val_npz processed/mitbih/val.npz --epochs 1 --batch 128 --out_dir outputs/teacher --debug
python -m src.train.train_erm --model student --train_npz processed/mitbih/train.npz --val_npz processed/mitbih/val.npz --epochs 1 --batch 128 --out_dir outputs/student --debug
```

### (D) Evaluate
```bash
python -m src.eval.evaluate --model student --ckpt outputs/student/best.pt --npz processed/mitbih/test.npz
python -m src.eval.evaluate --model student --ckpt outputs/student/best.pt --npz processed/incart/test.npz
```

### (E) Edge report
```bash
python -m src.tools.edge_report --model student --input_len 256
```

---

## 13) One-command pipeline script

Implement `scripts/run_all.sh` to:
- env check
- dataset check
- preprocess both datasets
- train teacher (short run)
- train student baseline + augmented
- evaluate on both test sets
- print a final summary

---

## 14) Definition of “done” (acceptance criteria)

Project is “working” when:
1. Preprocessing generates `.npz` datasets without errors.
2. Teacher and student both train for at least 1 epoch and save checkpoints.
3. Evaluation outputs macro-F1 and confusion matrix for both MIT‑BIH DS2 and INCART test.
4. `edge_report` confirms student is materially smaller than teacher.
5. Student+Aug shows improved INCART macro-F1 over Student baseline OR results are transparently reported with discussion.

---

If you follow this spec, Codex can implement the entire pipeline in a runnable, verifiable way aligned with edge constraints.
