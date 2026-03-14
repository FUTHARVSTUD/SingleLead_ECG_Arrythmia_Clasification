from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.configs.splits import (
    MITBIH_DSC_TEST,
    MITBIH_DSC_TRAIN,
    MITBIH_DS1,
    MITBIH_DS2,
    MITBIH_RECORDS,
)
from src.data import aami
from src.data.wfdb_io import (
    extract_windows,
    filter_annotations,
    normalize_signal,
    read_wfdb_record,
    resample_signal,
)

CHANNEL_PREFERENCE = ("MLII", "II", "V1", "V2")


def _parse_record_list(value: str) -> List[str]:
    if not value:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


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
        if args.context_beats > 1:
            return np.zeros((0, args.context_beats, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.zeros((0, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
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
        if args.context_beats > 1:
            return np.zeros((0, args.context_beats, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.zeros((0, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(samples).astype(np.float32), np.asarray(labels, dtype=np.int64)


def _collect_records(records: List[str], args: argparse.Namespace) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for rec in tqdm(records, desc="MIT-BIH records"):
        x_rec, y_rec = _collect_record(rec, args)
        if len(x_rec) == 0:
            continue
        data[rec] = (x_rec, y_rec)
    if not data:
        raise RuntimeError("No beats collected; verify WFDB files are present and valid")
    return data


def _record_counts(record_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
    counts = {}
    for rec, (_, labels) in record_data.items():
        counts[rec] = np.bincount(labels, minlength=len(aami.AAMI_CLASSES))
    return counts


def _select_val_records(
    record_counts: Dict[str, np.ndarray],
    val_ratio: float,
    focus_classes: List[int],
) -> List[str]:
    records = list(record_counts.keys())
    if not records:
        return []
    val_count = max(1, int(len(records) * val_ratio))
    if val_count >= len(records):
        val_count = max(1, len(records) - 1)
    selected: List[str] = []
    for cls in focus_classes:
        candidates = [r for r in records if r not in selected and record_counts[r][cls] > 0]
        if candidates:
            rec = min(candidates, key=lambda r: record_counts[r][cls])
            selected.append(rec)
            if len(selected) >= val_count:
                return selected
    remaining = [r for r in records if r not in selected]
    remaining.sort(
        key=lambda r: (
            sum(record_counts[r][cls] for cls in focus_classes) if focus_classes else 0,
            record_counts[r].sum(),
        )
    )
    selected.extend(remaining[: max(0, val_count - len(selected))])
    return selected


def main():
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH NPZ splits")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--target_fs", type=int, default=250)
    parser.add_argument("--window_len", type=int, default=256)
    parser.add_argument("--context_beats", type=int, default=1)
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of DS1 records reserved for validation (patient-level split)",
    )
    parser.add_argument(
        "--train_records",
        type=str,
        default="",
        help="Explicit comma-separated list of records used for training",
    )
    parser.add_argument(
        "--test_records",
        type=str,
        default="",
        help="Explicit comma-separated list of records used for testing",
    )
    parser.add_argument("--val_records", type=str, default="", help="Explicit comma-separated validation records")
    parser.add_argument(
        "--val_focus_classes",
        type=str,
        default="3,4",
        help="Class indices prioritized when auto-selecting validation records",
    )
    parser.add_argument(
        "--extra_train_records",
        type=str,
        default="",
        help="Additional record IDs appended to the training set",
    )
    parser.add_argument(
        "--minority_classes",
        type=str,
        default="",
        help="Minority class indices to augment within the training split",
    )
    parser.add_argument(
        "--minority_factor",
        type=int,
        default=0,
        help="Number of noisy copies added per minority-class beat",
    )
    parser.add_argument(
        "--minority_noise",
        type=float,
        default=0.01,
        help="Gaussian noise scale for minority augmentation",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    default_train = MITBIH_DSC_TRAIN if MITBIH_DSC_TRAIN else MITBIH_DS1
    default_test = MITBIH_DSC_TEST if MITBIH_DSC_TEST else MITBIH_DS2

    train_pool = _parse_record_list(args.train_records) or default_train
    test_records = _parse_record_list(args.test_records) or default_test

    def _validate_records(records: List[str], label: str):
        missing = [rec for rec in records if rec not in MITBIH_RECORDS]
        if missing:
            raise RuntimeError(f"Unknown {label} records: {missing}")

    _validate_records(train_pool, "train")
    _validate_records(test_records, "test")

    train_pool = [rec for rec in train_pool if rec not in test_records]
    if not train_pool:
        raise RuntimeError("No training records remain after removing test overlap")

    train_data = _collect_records(train_pool, args)
    record_counts = _record_counts(train_data)
    focus_classes = _parse_class_list(args.val_focus_classes)
    if args.val_records:
        val_records = [rec for rec in _parse_record_list(args.val_records) if rec in train_data]
        if not val_records:
            raise RuntimeError("Provided --val_records not found among training records")
    else:
        val_records = _select_val_records(record_counts, args.val_ratio, focus_classes)
    train_records = [rec for rec in train_data.keys() if rec not in val_records]
    if not train_records:
        raise RuntimeError("Validation split consumed all training records; reduce --val_ratio")

    extra_candidates = _parse_record_list(args.extra_train_records)
    extra_records = [rec for rec in extra_candidates if rec not in train_data and rec not in test_records]
    if extra_records:
        _validate_records(extra_records, "extra train")
    extra_data = _collect_records(extra_records, args) if extra_records else {}

    def _concat(selected: List[str], pool: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        if not selected:
            return np.zeros((0, args.window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        xs, ys = zip(*(pool[rec] for rec in selected))
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    train_x, train_y = _concat(train_records, train_data)
    if extra_data:
        extra_x, extra_y = _concat(list(extra_data.keys()), extra_data)
        train_x = np.concatenate([train_x, extra_x], axis=0)
        train_y = np.concatenate([train_y, extra_y], axis=0)

    minority_classes = _parse_class_list(args.minority_classes)
    if minority_classes and args.minority_factor > 0:
        train_x, train_y = _augment_minority(
            train_x, train_y, minority_classes, args.minority_factor, args.minority_noise
        )

    val_x, val_y = _concat(val_records, train_data)

    test_data = _collect_records(test_records, args)
    test_x, test_y = _concat(list(test_data.keys()), test_data)

    def _save(path: Path, x: np.ndarray, y: np.ndarray):
        np.savez_compressed(path, x=x, y=y)
        return {"path": str(path), "examples": len(x)}

    summary = {
        "train": _save(Path(args.out_dir) / "train.npz", train_x, train_y),
        "val": _save(Path(args.out_dir) / "val.npz", val_x, val_y),
        "test": _save(Path(args.out_dir) / "test.npz", test_x, test_y),
        "train_records": train_records + list(extra_data.keys()),
        "val_records": val_records,
        "extra_train_records": list(extra_data.keys()),
        "test_records": test_records,
        "minority_classes": minority_classes,
        "minority_factor": args.minority_factor,
    }

    (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
