"""Dynamic CNN model that materializes architectures from NAS configs."""

from __future__ import annotations

from typing import Any, Dict

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
    """Squeeze-and-Excitation block for channel-wise feature recalibration.

    Applies global average pooling to squeeze spatial information, then uses
    two fully-connected layers to produce per-channel attention weights that
    are multiplied back onto the feature map (excitation).

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio for the bottleneck FC layer (default 16).
    """

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
        scale = self.squeeze(x)                    # (N, C, 1, 1)
        scale = self.excitation(scale)             # (N, C)
        scale = scale.view(scale.size(0), -1, 1, 1)  # (N, C, 1, 1)
        return x * scale


class DynamicCNN(nn.Module):
    """Constructs a variable-depth CNN from architecture configuration.

    Layer blocks follow Conv2d -> BatchNorm? -> Activation -> SE? -> Dropout? -> Pool.
    Optional skip connections are applied as residual additions with channel
    alignment and spatial resizing for robustness across variable depths.
    """

    def __init__(self, arch_config: Dict[str, Any], num_classes: int) -> None:
        """Builds a dynamic CNN from architecture dictionary."""
        super().__init__()
        self.arch_config = arch_config
        self.num_layers = int(arch_config["num_layers"])
        self.use_skip_connections = bool(arch_config["use_skip_connections"])
        self.use_se_blocks = bool(arch_config.get("use_se_blocks", False))

        in_channels = 3
        self.blocks = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            out_channels = int(arch_config["filters"][layer_idx])
            kernel_size = int(arch_config["kernels"][layer_idx])
            padding = kernel_size // 2

            block = nn.ModuleDict(
                {
                    "conv": nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    "act": _activation(str(arch_config["activation"])),
                    "pool": (
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        if arch_config["pooling"] == "max"
                        else nn.AvgPool2d(kernel_size=2, stride=2)
                    ),
                }
            )
            if arch_config["use_batchnorm"]:
                block["bn"] = nn.BatchNorm2d(out_channels)
            if self.use_se_blocks:
                block["se"] = SEBlock(out_channels)
            if arch_config["use_dropout"]:
                block["drop"] = nn.Dropout2d(float(arch_config["dropout_rate"]))

            self.blocks.append(block)
            in_channels = out_channels

        # Learned 1x1 projections for skip connections, one per inter-block transition.
        # Each projection maps filters[i] -> filters[i+1] so the residual channel count
        # always matches the block output without zero-padding or feature slicing.
        if self.use_skip_connections:
            in_chs = [int(arch_config["filters"][i]) for i in range(self.num_layers - 1)]
            out_chs = [int(arch_config["filters"][i]) for i in range(1, self.num_layers)]
            self.skip_projs = nn.ModuleList([
                nn.Conv2d(ic, oc, kernel_size=1, bias=False) if ic != oc else nn.Identity()
                for ic, oc in zip(in_chs, out_chs)
            ])
        else:
            self.skip_projs = nn.ModuleList()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    @staticmethod
    def _match_channels(x: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Matches tensor channel size by slicing or zero-padding.

        Args:
            x: Input tensor with shape (N, C, H, W).
            target_channels: Desired channel dimension.

        Returns:
            torch.Tensor: Tensor with updated channel size.
        """
        current_channels = x.shape[1]
        if current_channels == target_channels:
            return x
        if current_channels > target_channels:
            return x[:, :target_channels, :, :]

        pad_channels = target_channels - current_channels
        pad_shape = (x.shape[0], pad_channels, x.shape[2], x.shape[3])
        padding = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
        return torch.cat([x, padding], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass through the dynamic CNN.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (N, num_classes).
        """
        out = x
        for layer_idx, block in enumerate(self.blocks):
            residual = out
            out = block["conv"](out)
            if "bn" in block:
                out = block["bn"](out)
            out = block["act"](out)
            if "se" in block:
                out = block["se"](out)
            if "drop" in block:
                out = block["drop"](out)
            out = block["pool"](out)

            if self.use_skip_connections and layer_idx > 0:
                resized = F.adaptive_avg_pool2d(residual, output_size=out.shape[-2:])
                projected = self.skip_projs[layer_idx - 1](resized)
                out = out + projected

        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits

