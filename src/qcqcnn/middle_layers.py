"""
MiddleLayerBase abstraction + 4 parameter-matched variants

The quantum conv filter and the VQC tail stay exactly as in the original
QCQ-CNN repo. Only the classical "middle" block is swapped. Every variant
must:
  1. Subclass MiddleLayerBase and implement forward().
  2. Take the same in_channels and produce the same out_channels/out_spatial
     shape, so the VQC input contract never changes.
  3. Land within +/- ~5% of BASELINE_PARAM_COUNT (checked at the bottom of
     this file) so the factorial experiment isolates architecture motif,
     not capacity.

"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class MiddleLayerBase(nn.Module, ABC):
    """Common contract for all middle-layer variants."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineCNN(MiddleLayerBase):
    """Control condition: the original paper's shallow CNN, unmodified."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 16):
        super().__init__(in_channels, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualMiddleLayer(MiddleLayerBase):
    """Variant R: single residual block, ResNet motif."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 15):
        super().__init__(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # NOTE:
        # 1x1 projection so the skip path matches out_channels even when
        # in_channels != out_channels.
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class DenseMiddleLayer(MiddleLayerBase):
    """Variant D: 3 conv layers with concatenated feature reuse, DenseNet motif."""

    def __init__(self, in_channels: int, out_channels: int, growth: int = 6):
        super().__init__(in_channels, out_channels)
        c0 = in_channels
        self.conv1 = nn.Conv2d(c0, growth, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c0 + growth, growth, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c0 + 2 * growth, growth, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c0 + 3 * growth)
        # NOTE:
        # Final 1x1 to project the concatenated feature stack down to
        # out_channels, keeping the output contract identical to the others.
        self.project = nn.Conv2d(c0 + 3 * growth, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f1 = self.relu(self.conv1(x))
        x1 = torch.cat([x, f1], dim=1)
        f2 = self.relu(self.conv2(x1))
        x2 = torch.cat([x1, f2], dim=1)
        f3 = self.relu(self.conv3(x2))
        x3 = torch.cat([x2, f3], dim=1)
        return self.relu(self.project(self.bn(x3)))


class InceptionMiddleLayer(MiddleLayerBase):
    """Variant I: factorized (1x3 + 3x1) depthwise-separable convs, Inception motif."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 112):
        super().__init__(in_channels, out_channels)
        # NOTE:
        # Depthwise factorized conv: 1x3 then 3x1, done depthwise then
        # pointwise-mixed, which is what keeps this cheaper than a plain
        # 3x3 conv at the same width.
        self.depthwise1x3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1),
            groups=in_channels,
        )
        self.depthwise3x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0),
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, width, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.out_conv = nn.Conv2d(width, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise1x3(x)
        x = self.depthwise3x1(x)
        x = self.relu(self.bn1(self.pointwise(x)))
        return self.relu(self.bn2(self.out_conv(x)))


MIDDLE_LAYER_REGISTRY = {
    "baseline": BaselineCNN,
    "residual": ResidualMiddleLayer,
    "dense": DenseMiddleLayer,
    "inception": InceptionMiddleLayer,
}


def build_middle_layer(name: str, in_channels: int, out_channels: int) -> MiddleLayerBase:
    """Factory so the training loop never branches on architecture by name."""
    if name not in MIDDLE_LAYER_REGISTRY:
        raise ValueError(f"Unknown middle layer '{name}', choose from {list(MIDDLE_LAYER_REGISTRY)}")
    return MIDDLE_LAYER_REGISTRY[name](in_channels, out_channels)
