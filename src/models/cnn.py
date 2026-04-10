"""Dynamic CNN model with ResNet-style bottleneck blocks for RA-NAS.

Each NAS stage is a stack of ResNet bottleneck residual blocks.
Architecture dimensions (num_stages, filters, width_multiplier, block_depth)
are all NAS-searchable. This design is capable of reaching ~95%+ on CIFAR-10.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    """Returns activation module by name."""
    mapping = {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[name]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise recalibration."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        bottleneck = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale.view(scale.size(0), -1, 1, 1)


class ResBottleneckBlock(nn.Module):
    """Pre-activation ResNet bottleneck: BN→Act→Conv(1)→BN→Act→Conv(3)→BN→Act→Conv(1) + skip.

    Uses pre-activation (He et al. 2016 v2) for better gradient flow.
    The bottleneck ratio compresses channels by 4 then expands back.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: str,
        use_se: bool,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        mid_channels = max(1, out_channels // self.expansion)
        padding = kernel_size // 2

        self.bn1   = nn.BatchNorm2d(in_channels)
        self.act1  = _activation(activation)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        self.bn2   = nn.BatchNorm2d(mid_channels)
        self.act2  = _activation(activation)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)

        self.bn3   = nn.BatchNorm2d(mid_channels)
        self.act3  = _activation(activation)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        self.se = SEBlock(out_channels) if use_se else None
        self.drop = nn.Dropout2d(dropout_rate) if dropout_rate > 0.0 else None

        # Projection shortcut when spatial or channel dimensions change
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        out = self.conv3(self.act3(self.bn3(out)))
        if self.se is not None:
            out = self.se(out)
        if self.drop is not None:
            out = self.drop(out)
        return out + self.shortcut(x)


class DynamicCNN(nn.Module):
    """Variable-depth ResNet built from NAS architecture config.

    The network is organised as:
        Stem (3x3 conv, 64ch, BN, Act)
        → N stages, each a stack of ResBottleneckBlocks
        → Global average pool
        → Linear classifier

    Each stage may use a different filter width, kernel size, and block count.
    Downsampling by stride=2 happens at the first block of each stage after stage 0.
    """

    def __init__(self, arch_config: Dict[str, Any], num_classes: int) -> None:
        super().__init__()
        self.arch_config = arch_config
        num_stages = int(arch_config["num_layers"])   # "layers" = number of stages
        filters: List[int] = [int(f) for f in arch_config["filters"]]
        kernels: List[int] = [int(k) for k in arch_config["kernels"]]
        block_depths: List[int] = [int(d) for d in arch_config.get(
            "block_depths", [2] * num_stages)]
        activation = str(arch_config["activation"])
        use_se = bool(arch_config.get("use_se_blocks", False))
        dropout_rate = float(arch_config["dropout_rate"]) if arch_config["use_dropout"] else 0.0

        # ── Stem: process 32x32 CIFAR without downsampling ──────────────────
        stem_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            _activation(activation),
        )

        # ── Stages ───────────────────────────────────────────────────────────
        in_ch = stem_channels
        stages: List[nn.Module] = []
        for stage_idx in range(num_stages):
            out_ch = filters[stage_idx]
            k = kernels[stage_idx]
            depth = block_depths[stage_idx]
            stride = 1 if stage_idx == 0 else 2   # downsample at stage boundary

            blocks: List[nn.Module] = []
            for blk_idx in range(depth):
                blk_stride = stride if blk_idx == 0 else 1
                blk_in = in_ch if blk_idx == 0 else out_ch
                blocks.append(ResBottleneckBlock(
                    in_channels=blk_in,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=blk_stride,
                    activation=activation,
                    use_se=use_se,
                    dropout_rate=dropout_rate,
                ))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.stages = nn.ModuleList(stages)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)

