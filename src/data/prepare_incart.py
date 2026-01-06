from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import wfdb
from tqdm import tqdm

from src.data import aami
from src.data.wfdb_io import extract_windows, filter_annotations, normalize_signal, resample_signal

CHANNEL_PREFERENCE = ("II", "MLII", "V1", "V2")
DEFAULT_PN_DIR = "incartdb/1.0.0"


def _pick_channel(sig_names: Sequence[str]) -> int:
    lookup = {name.upper(): idx for idx, name in enumerate(sig_names)}
    for name in CHANNEL_PREFERENCE:
        if name.upper() in lookup:
            return lookup[name.upper()]
    return 0


def _remote_record_name(record: str) -> str:
    # Local files are zero-padded like I0001; PhysioNet uses I01 naming.
    prefix = record[0]
    number = int(record[1:])
    return f"{prefix}{number:02d}"


def _load_annotations(record: str, data_dir: str, pn_dir: str) -> Tuple[np.ndarray, List[str]]:
    base_path = f"{data_dir}/{record}"
    try:
        ann = wfdb.rdann(base_path, "atr")
        return np.asarray(ann.sample, dtype=np.int64), list(ann.symbol)
    except Exception:
        pass
    remote_name = _remote_record_name(record)
    if pn_dir is None:
        raise FileNotFoundError(
            f"Annotation file for {record} not found locally and pn_dir disabled; cannot proceed"
        )
    ann = wfdb.rdann(remote_name, "atr", pn_dir=pn_dir)
    return np.asarray(ann.sample, dtype=np.int64), list(ann.symbol)


def _collect(record: str, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    sig, fields = wfdb.rdsamp(f"{args.data_dir}/{record}")
    channel_idx = _pick_channel(fields["sig_name"])
    lead = sig[:, channel_idx].astype(np.float32)
    lead = resample_signal(lead, int(fields["fs"]), args.target_fs)
    lead = normalize_signal(lead)
    ann_samples, symbols = _load_annotations(record, args.data_dir, args.pn_dir)
    ann_samples = np.round(ann_samples.astype(np.float32) * args.target_fs / fields["fs"]).astype(
        np.int64
    )
    ann_samples, symbols = filter_annotations(
        ann_samples, symbols, signal_len=len(lead), window_len=args.window_len
    )
    if len(ann_samples) == 0:
        return np.zeros((0, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    windows = extract_windows(lead, ann_samples, args.window_len)
    samples = []
    labels = []
    for win, symbol in zip(windows, symbols):
        label = aami.map_symbol(symbol)
        if label is None:
            continue
        samples.append(win)
        labels.append(label)
    if not samples:
        return np.zeros((0, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(samples).astype(np.float32), np.asarray(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Prepare INCART NPZ splits")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--target_fs", type=int, default=250)
    parser.add_argument("--window_len", type=int, default=256)
    parser.add_argument("--adapt_ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--pn_dir",
        type=str,
        default=DEFAULT_PN_DIR,
        help="PhysioNet directory for INCART annotations when local .atr files are absent",
    )
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    records = sorted(Path(args.data_dir).glob("I*.hea"))
    record_ids = [p.stem for p in records]
    if not record_ids:
        raise RuntimeError("No INCART records found")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(record_ids)
    adapt_count = max(1, int(len(record_ids) * args.adapt_ratio))
    adapt_records = record_ids[:adapt_count]
    test_records = record_ids[adapt_count:]
    if not test_records:
        # ensure we always have held-out test data
        test_records = adapt_records[adapt_count // 2 :]
        adapt_records = adapt_records[: adapt_count // 2]

    def _aggregate(rec_list: List[str]):
        xs, ys = [], []
        for rec in tqdm(rec_list, desc="INCART records"):
            x_rec, y_rec = _collect(rec, args)
            if len(x_rec) == 0:
                continue
            xs.append(x_rec)
            ys.append(y_rec)
        if not xs:
            return np.zeros((0, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    adapt_x, adapt_y = _aggregate(adapt_records)
    test_x, test_y = _aggregate(test_records)

    def _save(path: Path, x: np.ndarray, y: np.ndarray):
        np.savez_compressed(path, x=x, y=y)
        return {"path": str(path), "examples": len(x)}

    summary = {
        "adapt": _save(Path(args.out_dir) / "adapt.npz", adapt_x, adapt_y),
        "test": _save(Path(args.out_dir) / "test.npz", test_x, test_y),
    }
    (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
