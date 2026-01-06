"""Utilities for reading WFDB data and extracting beat windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import signal
import wfdb


@dataclass
class WfdbRecord:
    signal: np.ndarray  # shape (num_samples,)
    fs: int
    annotations: np.ndarray  # raw annotation indices (sample positions)
    symbols: List[str]


def _pick_channel(sig_names: Sequence[str], channel_preference: Sequence[str]) -> int:
    name_to_idx = {name.upper(): idx for idx, name in enumerate(sig_names)}
    for name in channel_preference:
        if name.upper() in name_to_idx:
            return name_to_idx[name.upper()]
    return 0


def read_wfdb_record(record: str, data_dir: str, channel_preference: Sequence[str]) -> WfdbRecord:
    path = f"{data_dir}/{record}"
    sig, fields = wfdb.rdsamp(path)
    if sig.ndim == 1:
        sig = sig[:, None]
    channel_idx = _pick_channel(fields["sig_name"], channel_preference)
    lead = sig[:, channel_idx]
    annotations = wfdb.rdann(path, "atr")
    return WfdbRecord(
        signal=lead.astype(np.float32),
        fs=int(fields["fs"]),
        annotations=np.asarray(annotations.sample, dtype=np.int64),
        symbols=list(annotations.symbol),
    )


def normalize_signal(sig: np.ndarray) -> np.ndarray:
    sig = sig.astype(np.float32)
    sig -= np.median(sig)
    std = np.std(sig) + 1e-6
    return sig / std


def resample_signal(sig: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    if orig_fs == target_fs:
        return sig.astype(np.float32)
    gcd = np.gcd(orig_fs, target_fs)
    up = target_fs // gcd
    down = orig_fs // gcd
    return signal.resample_poly(sig, up=up, down=down).astype(np.float32)


def extract_windows(sig: np.ndarray, annotation_samples: np.ndarray, window_len: int) -> np.ndarray:
    half = window_len // 2
    padded = np.pad(sig, (half, half), mode="constant")
    centers = annotation_samples + half
    windows = [padded[c - half : c + half] for c in centers]
    return np.stack(windows, axis=0)


def filter_annotations(
    annotation_samples: np.ndarray,
    symbols: Sequence[str],
    signal_len: int,
    window_len: int,
) -> Tuple[np.ndarray, List[str]]:
    half = window_len // 2
    keep_idx: List[int] = []
    keep_symbols: List[str] = []
    for idx, sample in enumerate(annotation_samples):
        if sample - half < 0 or sample + half >= signal_len:
            continue
        keep_idx.append(idx)
        keep_symbols.append(symbols[idx])
    if not keep_idx:
        return np.zeros(0, dtype=np.int64), []
    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    return annotation_samples[keep_idx_arr], keep_symbols
