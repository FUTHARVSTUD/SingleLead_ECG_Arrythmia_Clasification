from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from src.data import aami
from src.data.wfdb_io import read_wfdb_record, resample_signal, normalize_signal, filter_annotations, extract_windows

CHANNEL_PREFERENCE = ("MLII", "II", "V1", "V2")


def _collect_record(record: str, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    wfdb_record = read_wfdb_record(record, args.data_dir, CHANNEL_PREFERENCE)
    signal = resample_signal(wfdb_record.signal, wfdb_record.fs, args.target_fs)
    signal = normalize_signal(signal)
    ann_samples = np.round(
        wfdb_record.annotations.astype(np.float32) * args.target_fs / wfdb_record.fs
    ).astype(np.int64)
    ann_samples, symbols = filter_annotations(
        ann_samples, wfdb_record.symbols, signal_len=len(signal), window_len=args.window_len
    )
    if len(ann_samples) == 0:
        shape = (0, args.context_beats, args.window_len) if args.context_beats > 1 else (0, args.window_len)
        return np.zeros(shape, dtype=np.float32), np.zeros((0,), dtype=np.int64)
    windows = extract_windows(signal, ann_samples, args.window_len, context_beats=args.context_beats)
    samples = []
    labels = []
    for win, symbol in zip(windows, symbols):
        label = aami.map_symbol(symbol)
        if label is None:
            continue
        samples.append(win)
        labels.append(label)
    if not samples:
        shape = (0, args.context_beats, args.window_len) if args.context_beats > 1 else (0, args.window_len)
        return np.zeros(shape, dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(samples).astype(np.float32), np.asarray(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Prepare CU-VT NPZ splits")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_npz", type=str, required=True)
    parser.add_argument("--summary", type=str, default=None)
    parser.add_argument("--target_fs", type=int, default=250)
    parser.add_argument("--window_len", type=int, default=256)
    parser.add_argument("--context_beats", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    records = sorted(p.stem for p in data_dir.glob("*.hea"))
    if not records:
        raise RuntimeError("No CU-VT records found")

    xs, ys = [], []
    for rec in tqdm(records, desc="CU-VT records"):
        x_rec, y_rec = _collect_record(rec, args)
        if len(x_rec) == 0:
            continue
        xs.append(x_rec)
        ys.append(y_rec)
    if not xs:
        raise RuntimeError("No beats extracted from CU-VT dataset")

    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, x=X, y=Y)
    summary = {"path": str(out_path), "examples": int(len(Y)), "context_beats": int(args.context_beats)}
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
