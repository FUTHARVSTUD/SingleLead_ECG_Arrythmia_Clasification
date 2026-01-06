from __future__ import annotations

from typing import List

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 5, base_channels: int = 32, layers: List[int] | None = None):
        super().__init__()
        layers = layers or [2, 2, 2, 2]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        channels = base_channels
        self.stages = nn.ModuleList()
        for i, num_blocks in enumerate(layers):
            out_channels = base_channels * (2**i)
            stride = 1 if i == 0 else 2
            stage = self._make_layer(channels, out_channels, num_blocks, stride)
            self.stages.append(stage)
            channels = out_channels
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, num_classes))

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        modules = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            modules.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x


def resnet1d_teacher(**kwargs) -> ResNet1D:
    return ResNet1D(**kwargs)
