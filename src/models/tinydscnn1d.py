from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class TinyDSCNN1D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 5, base_channels: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        channels = base_channels
        layers = []
        for stride, mult in zip([1, 2, 1, 2], [1, 2, 2, 2]):
            out_channels = channels * mult
            layers.append(DepthwiseSeparableConv(channels, out_channels, kernel_size=5, stride=stride))
            channels = out_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x


def tinydscnn1d_student(**kwargs) -> TinyDSCNN1D:
    return TinyDSCNN1D(**kwargs)
