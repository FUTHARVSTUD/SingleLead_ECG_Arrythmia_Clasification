from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.configs.splits import MITBIH_DS1, MITBIH_DS2
from src.data import aami
from src.data.wfdb_io import (
    extract_windows,
    filter_annotations,
    normalize_signal,
    read_wfdb_record,
    resample_signal,
)

CHANNEL_PREFERENCE = ("MLII", "II", "V1", "V2")


def _collect_beats(records: List[str], args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    windows = []
    labels = []
    for rec in tqdm(records, desc="MIT-BIH records"):
        wfdb_record = read_wfdb_record(rec, args.data_dir, CHANNEL_PREFERENCE)
        signal = resample_signal(wfdb_record.signal, wfdb_record.fs, args.target_fs)
        signal = normalize_signal(signal)
        ann_samples = np.round(
            wfdb_record.annotations.astype(np.float32) * args.target_fs / wfdb_record.fs
        ).astype(np.int64)
        ann_samples, symbols = filter_annotations(
            ann_samples, wfdb_record.symbols, signal_len=len(signal), window_len=args.window_len
        )
        if len(ann_samples) == 0:
            continue
        beat_windows = extract_windows(signal, ann_samples, args.window_len)
        for win, symbol in zip(beat_windows, symbols):
            label = aami.map_symbol(symbol)
            if label is None:
                continue
            windows.append(win)
            labels.append(label)
    if not windows:
        raise RuntimeError("No beats collected; verify WFDB files are present and valid")
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int64)


def _split_train_val(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    val_size = max(1, int(len(x) * val_ratio))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def main():
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH NPZ splits")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--target_fs", type=int, default=250)
    parser.add_argument("--window_len", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ds1_x, ds1_y = _collect_beats(MITBIH_DS1, args)
    train_x, train_y, val_x, val_y = _split_train_val(ds1_x, ds1_y, args.val_ratio, args.seed)

    ds2_x, ds2_y = _collect_beats(MITBIH_DS2, args)

    def _save(path: Path, x: np.ndarray, y: np.ndarray):
        np.savez_compressed(path, x=x, y=y)
        return {"path": str(path), "examples": len(x)}

    summary = {
        "train": _save(Path(args.out_dir) / "train.npz", train_x, train_y),
        "val": _save(Path(args.out_dir) / "val.npz", val_x, val_y),
        "test": _save(Path(args.out_dir) / "test.npz", ds2_x, ds2_y),
    }

    (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
