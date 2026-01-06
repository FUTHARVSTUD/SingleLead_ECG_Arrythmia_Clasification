from __future__ import annotations

from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader


def paired_iterator(loader_a: DataLoader, loader_b: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches from loader_a and loader_b sequentially.

    The shorter loader defines the number of iterations to avoid StopIteration issues.
    """
    for (xa, ya), (xb, yb) in zip(loader_a, loader_b):
        yield (xa, ya), (xb, yb)
