from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzDataset(Dataset):
    def __init__(self, path: str, augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        data = np.load(path)
        self.x = torch.from_numpy(data["x"]).float().unsqueeze(1)
        self.y = torch.from_numpy(data["y"]).long()
        self.augment = augment
        self.path = str(path)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y[idx]

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item() + 1)

    def class_counts(self):
        counts = torch.bincount(self.y, minlength=self.num_classes)
        return counts.cpu().numpy()
