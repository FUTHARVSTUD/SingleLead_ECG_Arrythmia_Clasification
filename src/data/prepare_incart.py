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


def _parse_class_list(value: str) -> List[int]:
    if not value:
        return []
    classes: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        classes.append(int(token))
    return classes


def _augment_minority(
    x: np.ndarray,
    y: np.ndarray,
    class_ids: List[int],
    factor: int,
    noise_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if factor <= 0 or not class_ids or len(x) == 0:
        return x, y
    rng = np.random.default_rng(0)
    augmented = [x]
    targets = [y]
    for cls in class_ids:
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        samples = x[idx]
        for _ in range(factor):
            noise = rng.normal(loc=0.0, scale=noise_scale, size=samples.shape).astype(np.float32)
            augmented.append(samples + noise)
            targets.append(np.full(len(samples), cls, dtype=np.int64))
    if len(augmented) == 1:
        return x, y
    return np.concatenate(augmented, axis=0), np.concatenate(targets, axis=0)


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
        if args.context_beats > 1:
            return np.zeros((0, args.context_beats, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.zeros((0, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    windows = extract_windows(lead, ann_samples, args.window_len, context_beats=args.context_beats)
    samples = []
    labels = []
    for win, symbol in zip(windows, symbols):
        label = aami.map_symbol(symbol)
        if label is None:
            continue
        samples.append(win)
        labels.append(label)
    if not samples:
        if args.context_beats > 1:
            return np.zeros((0, args.context_beats, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.zeros((0, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(samples).astype(np.float32), np.asarray(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Prepare INCART NPZ splits")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--target_fs", type=int, default=250)
    parser.add_argument("--window_len", type=int, default=256)
    parser.add_argument("--context_beats", type=int, default=1)
    parser.add_argument("--adapt_ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--adapt_val_ratio",
        type=float,
        default=0.1,
        help="Portion of the adapt split reserved as validation for fine-tuning",
    )
    parser.add_argument(
        "--pn_dir",
        type=str,
        default=DEFAULT_PN_DIR,
        help="PhysioNet directory for INCART annotations when local .atr files are absent",
    )
    parser.add_argument(
        "--minority_classes",
        type=str,
        default="",
        help="Class indices to augment within INCART adapt_train",
    )
    parser.add_argument(
        "--minority_factor",
        type=int,
        default=0,
        help="Number of noisy copies per minority-class beat in adapt_train",
    )
    parser.add_argument(
        "--minority_noise",
        type=float,
        default=0.01,
        help="Gaussian noise scale used for INCART minority augmentation",
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

    def _split_samples(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int):
        if len(x) == 0 or val_ratio <= 0:
            return (x, y), (np.zeros((0, x.shape[-1]), dtype=x.dtype), np.zeros((0,), dtype=y.dtype))
        rng = np.random.default_rng(seed)
        idx = np.arange(len(x))
        rng.shuffle(idx)
        val_size = max(1, int(len(x) * val_ratio))
        if val_size >= len(x):
            val_size = len(x) // 2
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]
        return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])

    (adapt_train_x, adapt_train_y), (adapt_val_x, adapt_val_y) = _split_samples(
        adapt_x, adapt_y, args.adapt_val_ratio, args.seed
    )

    minority_classes = _parse_class_list(args.minority_classes)
    if minority_classes and args.minority_factor > 0:
        adapt_train_x, adapt_train_y = _augment_minority(
            adapt_train_x, adapt_train_y, minority_classes, args.minority_factor, args.minority_noise
        )

    def _save(path: Path, x: np.ndarray, y: np.ndarray):
        np.savez_compressed(path, x=x, y=y)
        return {"path": str(path), "examples": len(x)}

    summary = {
        "adapt": _save(Path(args.out_dir) / "adapt.npz", adapt_x, adapt_y),
        "adapt_train": _save(Path(args.out_dir) / "adapt_train.npz", adapt_train_x, adapt_train_y),
        "adapt_val": _save(Path(args.out_dir) / "adapt_val.npz", adapt_val_x, adapt_val_y),
        "test": _save(Path(args.out_dir) / "test.npz", test_x, test_y),
        "minority_classes": minority_classes,
        "minority_factor": args.minority_factor,
    }
    (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
