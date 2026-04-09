"""Evaluation metric utilities used across the RA-NAS pipeline."""

from __future__ import annotations

import time
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn


def accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    """Computes top-k accuracy percentages.

    Args:
        outputs: Model logits tensor with shape (N, C).
        targets: Ground-truth labels tensor with shape (N,).
        topk: Tuple of k-values for top-k metrics.

    Returns:
        Dict[str, float]: Mapping like {"top1": value, "top5": value}.
    """
    if outputs.ndim != 2:
        raise ValueError(f"outputs must be rank-2 logits, got shape {tuple(outputs.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be rank-1 labels, got shape {tuple(targets.shape)}")

    max_k = min(max(topk), outputs.size(1))
    _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    expanded_targets = targets.view(1, -1).expand_as(pred)
    correct = pred.eq(expanded_targets)

    metrics: Dict[str, float] = {}
    batch_size = targets.size(0)
    for k in topk:
        used_k = min(k, outputs.size(1))
        correct_k = correct[:used_k].reshape(-1).float().sum(0)
        metrics[f"top{k}"] = float(correct_k.item() * 100.0 / max(1, batch_size))
    return metrics


def average_loss(losses: Iterable[float]) -> float:
    """Computes average loss from an iterable.

    Args:
        losses: Loss values.

    Returns:
        float: Mean value or 0.0 if empty.
    """
    values = list(losses)
    if not values:
        return 0.0
    return float(np.mean(values))


def parameter_count(model: nn.Module) -> int:
    """Counts trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def inference_time(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: str,
    n_runs: int = 50,
) -> float:
    """Measures average inference latency over multiple runs in milliseconds.

    Args:
        model: PyTorch model.
        input_size: Input shape tuple (N, C, H, W).
        device: Device string.
        n_runs: Number of timed runs.

    Returns:
        float: Mean inference time per run in milliseconds.
    """
    dummy = torch.randn(*input_size, device=device)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

    if was_training:
        model.train()
    return float(np.mean(timings))


def flops_estimate(model: nn.Module, input_size: Tuple[int, int, int, int]) -> int:
    """Estimates model FLOPs using Conv2d and Linear operation counts.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape (N, C, H, W).

    Returns:
        int: Approximate FLOPs for one forward pass.
    """
    flops = {"value": 0}
    hooks = []

    def conv_hook(module: nn.Conv2d, _inputs, outputs: torch.Tensor) -> None:
        batch_size, out_channels, out_h, out_w = outputs.shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
        bias_ops = 1 if module.bias is not None else 0
        flops["value"] += int(batch_size * out_channels * out_h * out_w * (kernel_ops + bias_ops))

    def linear_hook(module: nn.Linear, _inputs, outputs: torch.Tensor) -> None:
        batch_size = outputs.shape[0]
        flops["value"] += int(batch_size * module.in_features * module.out_features)

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        _ = model(torch.randn(*input_size, device=device))
    if was_training:
        model.train()

    for handle in hooks:
        handle.remove()
    return int(flops["value"])
