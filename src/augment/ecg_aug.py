from __future__ import annotations

import math
from typing import Optional

import torch


class RampScheduler:
    def __init__(self, warmup: int = 5, mid: int = 10):
        self.warmup = warmup
        self.mid = mid
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def value(self) -> float:
        if self.epoch < self.warmup:
            return 0.0
        if self.epoch < self.mid:
            return 0.5
        return 1.0


class ECGAugmentor:
    def __init__(self, sample_rate: int = 250, scheduler: Optional[RampScheduler] = None):
        self.sample_rate = sample_rate
        self.scheduler = scheduler or RampScheduler()

    def set_epoch(self, epoch: int):
        if self.scheduler:
            self.scheduler.set_epoch(epoch)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.scheduler and self.scheduler.value() == 0:
            return x
        strength = self.scheduler.value()
        y = x.clone()
        std = y.std()
        if std == 0:
            std = torch.tensor(1.0, dtype=y.dtype, device=y.device)

        if torch.rand(1).item() < 0.7:
            y = self._baseline_wander(y, std, strength)
        if torch.rand(1).item() < 0.6:
            y = self._powerline_hum(y, std, strength)
        if torch.rand(1).item() < 0.8:
            y = self._gaussian_noise(y, std, strength)
        if torch.rand(1).item() < 0.7:
            y = self._amplitude_scale(y, strength)
        if torch.rand(1).item() < 0.5:
            y = self._time_shift(y, strength)
        return y

    def _baseline_wander(self, x: torch.Tensor, std: torch.Tensor, strength: float):
        length = x.shape[-1]
        t = torch.arange(length, device=x.device) / self.sample_rate
        freq = torch.empty(1).uniform_(0.05, 0.3).item()
        amp = 0.05 * strength * std
        sine = amp * torch.sin(2 * math.pi * freq * t)
        return x + sine

    def _powerline_hum(self, x: torch.Tensor, std: torch.Tensor, strength: float):
        length = x.shape[-1]
        t = torch.arange(length, device=x.device) / self.sample_rate
        freq = 50.0
        amp = 0.02 * strength * std
        sine = amp * torch.sin(2 * math.pi * freq * t)
        return x + sine

    def _gaussian_noise(self, x: torch.Tensor, std: torch.Tensor, strength: float):
        noise = torch.randn_like(x) * (0.02 * strength) * std
        return x + noise

    def _amplitude_scale(self, x: torch.Tensor, strength: float):
        scale = 1.0 + 0.15 * strength * (torch.rand(1).item() * 2 - 1)
        return x * scale

    def _time_shift(self, x: torch.Tensor, strength: float):
        max_shift = max(1, int(x.shape[-1] * 0.02 * strength))
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=-1)


def augment_factory(sample_rate: int = 250) -> ECGAugmentor:
    return ECGAugmentor(sample_rate=sample_rate)
