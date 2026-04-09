"""Model builder and lightweight complexity utilities for RA-NAS."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from src.models.cnn import DynamicCNN


def build_model(arch_config: Dict[str, Any], num_classes: int, device: str) -> DynamicCNN:
    """Builds and moves a DynamicCNN instance to the target device.

    Args:
        arch_config: Architecture dictionary used to define model structure.
        num_classes: Number of classes in dataset labels.
        device: PyTorch device string.

    Returns:
        DynamicCNN: Instantiated model on target device.
    """
    model = DynamicCNN(arch_config=arch_config, num_classes=num_classes)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Counts trainable model parameters.

    Args:
        model: PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

