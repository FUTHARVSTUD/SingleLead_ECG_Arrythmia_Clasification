from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


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
    def __init__(
        self,
        sample_rate: int = 250,
        scheduler: Optional[RampScheduler] = None,
        warmup_epochs: int = 5,
        mid_epochs: int = 10,
        mixup_prob: float = 0.4,
        baseline_prob: float = 0.5,
        inversion_prob: float = 0.2,
        lead_drop_prob: float = 0.15,
        time_scale_prob: float = 0.4,
        segment_swap_prob: float = 0.3,
    ):
        self.sample_rate = sample_rate
        self.scheduler = scheduler or RampScheduler(warmup=warmup_epochs, mid=mid_epochs)
        self.mixup_prob = mixup_prob
        self.baseline_prob = baseline_prob
        self.inversion_prob = inversion_prob
        self.lead_drop_prob = lead_drop_prob
        self.time_scale_prob = time_scale_prob
        self.segment_swap_prob = segment_swap_prob
        self.class_buffers: dict[int, torch.Tensor] = {}

    def set_epoch(self, epoch: int):
        if self.scheduler:
            self.scheduler.set_epoch(epoch)

    def __call__(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.scheduler and self.scheduler.value() == 0:
            if label is not None:
                self._update_buffer(label, x)
            return x
        strength = self.scheduler.value()
        y = x.clone()
        std = y.std()
        if std == 0:
            std = torch.tensor(1.0, dtype=y.dtype, device=y.device)

        label_idx = None
        if label is not None:
            label_idx = int(label.item())
            if torch.rand(1).item() < self.mixup_prob:
                y = self._mixup_same_class(y, label_idx)

        if torch.rand(1).item() < 0.7:
            y = self._baseline_wander(y, std, strength)
        if torch.rand(1).item() < 0.6:
            y = self._powerline_hum(y, std, strength)
        if torch.rand(1).item() < 0.8:
            y = self._gaussian_noise(y, std, strength)
        if torch.rand(1).item() < self.baseline_prob:
            y = self._baseline_shift(y, std, strength)
        if torch.rand(1).item() < self.inversion_prob:
            y = self._invert_signal(y)
        if torch.rand(1).item() < 0.7:
            y = self._amplitude_scale(y, strength)
        if torch.rand(1).item() < 0.5:
            y = self._time_shift(y, strength)
        if torch.rand(1).item() < self.lead_drop_prob:
            y = self._lead_drop(y)
        if torch.rand(1).item() < self.time_scale_prob:
            y = self._time_scale(y, strength)
        if label_idx is not None and torch.rand(1).item() < self.segment_swap_prob:
            y = self._segment_swap(y, label_idx)

        if label_idx is not None:
            self._update_buffer(label_idx, x)
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

    def _baseline_shift(self, x: torch.Tensor, std: torch.Tensor, strength: float):
        offset = (torch.rand(1, device=x.device) * 2 - 1) * 0.1 * strength * std
        return x + offset

    def _invert_signal(self, x: torch.Tensor):
        return -x

    def _lead_drop(self, x: torch.Tensor):
        length = x.shape[-1]
        drop_len = max(int(length * torch.rand(1).item() * 0.6), int(length * 0.2))
        start = torch.randint(0, max(1, length - drop_len), (1,)).item()
        y = x.clone()
        y[..., start : start + drop_len] = 0.0
        return y

    def _time_scale(self, x: torch.Tensor, strength: float):
        length = x.shape[-1]
        scale = 1.0 + 0.2 * strength * (torch.rand(1).item() * 2 - 1)
        new_len = max(16, int(length * scale))
        x_in = x.unsqueeze(0)
        scaled = F.interpolate(x_in, size=new_len, mode="linear", align_corners=False)
        if new_len > length:
            start = torch.randint(0, new_len - length + 1, (1,)).item()
            scaled = scaled[..., start : start + length]
        elif new_len < length:
            pad = length - new_len
            left = pad // 2
            right = pad - left
            scaled = F.pad(scaled, (left, right))
        return scaled.squeeze(0)

    def _segment_swap(self, x: torch.Tensor, label: int):
        swap_sample = self.class_buffers.get(label)
        if swap_sample is None:
            return x
        length = x.shape[-1]
        seg_len = max(8, int(length * 0.1 * torch.rand(1).item()))
        start = torch.randint(0, max(1, length - seg_len), (1,)).item()
        swap = swap_sample.to(x.device)
        y = x.clone()
        y[..., start : start + seg_len] = swap[..., start : start + seg_len]
        return y

    def _update_buffer(self, label: int, signal: torch.Tensor):
        self.class_buffers[label] = signal.detach().cpu()

    def _mixup_same_class(self, x: torch.Tensor, label: int):
        mix_sample = self.class_buffers.get(label)
        if mix_sample is None:
            return x
        lam = torch.distributions.Beta(0.4, 0.4).sample().item()
        mix = mix_sample.to(x.device)
        return lam * x + (1 - lam) * mix


def augment_factory(sample_rate: int = 250) -> ECGAugmentor:
    return ECGAugmentor(sample_rate=sample_rate)
