from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzDataset(Dataset):
    def __init__(self, path: str, augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        data = np.load(path)
        arr = torch.from_numpy(data["x"]).float()
        if arr.ndim == 2:
            arr = arr.unsqueeze(1)
        elif arr.ndim != 3:
            raise ValueError(f"Unsupported input shape {arr.shape}; expected 2D or 3D array")
        self.x = arr
        self.y = torch.from_numpy(data["y"]).long()
        self.augment = augment
        self.path = str(path)
        self._in_channels = self.x.shape[1]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]
        if self.augment is not None:
            x = self.augment(x, y)
        return x, y

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item() + 1)

    @property
    def in_channels(self) -> int:
        return int(self._in_channels)

    def class_counts(self):
        counts = torch.bincount(self.y, minlength=self.num_classes)
        return counts.cpu().numpy()
