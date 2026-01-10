from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    class_counts = torch.bincount(labels, minlength=int(labels.max().item() + 1))
    weights = 1.0 / (class_counts.float() + 1e-6)
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Return normalized inverse-frequency weights for each class."""
    counts = torch.bincount(labels, minlength=int(labels.max().item() + 1)).float()
    counts = counts.clamp(min=1.0)
    weights = 1.0 / torch.sqrt(counts)
    weights /= weights.mean()
    return weights


def save_checkpoint(model: torch.nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_metrics(metrics: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best = -float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        if value > self.best:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
