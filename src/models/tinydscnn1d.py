from __future__ import annotations

import torch
from torch import nn


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class DepthwiseSeparableResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        use_se: bool = True,
    ):
        super().__init__()
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
        self.use_se = use_se
        self.se = SqueezeExcite(out_channels) if use_se else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.se(out)
        skip = self.skip(x)
        out = out + skip
        return self.act(out)


class TinyDSCNN1D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 5, base_channels: int = 24):
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
            layers.append(
                DepthwiseSeparableResBlock(
                    channels,
                    out_channels,
                    kernel_size=5,
                    stride=stride,
                    use_se=True,
                )
            )
            channels = out_channels
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        pooled = self.avgpool(x)
        flat = torch.flatten(pooled, 1)
        logits = self.fc(flat)
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        pooled = self.avgpool(x)
        flat = torch.flatten(pooled, 1)
        return flat

    def forward_with_features(self, x: torch.Tensor):
        feats = self.forward_features(x)
        logits = self.fc(feats)
        return logits, feats


def tinydscnn1d_student(**kwargs) -> TinyDSCNN1D:
    return TinyDSCNN1D(**kwargs)
