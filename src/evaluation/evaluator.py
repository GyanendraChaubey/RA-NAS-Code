"""Model evaluator for validation/test-time metrics in RA-NAS."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from src.evaluation.metrics import (
    accuracy,
    average_loss,
    flops_estimate,
    inference_time,
    parameter_count,
)


class Evaluator:
    """Runs model evaluation and returns a rich metric dictionary.

    Evaluator keeps inference-time metrics independent from training code,
    which supports clean ablation and analysis workflows.
    """

    def __init__(self, model: nn.Module, device: str) -> None:
        """Initializes evaluator with model and device.

        Args:
            model: Trained model instance.
            device: Device string.
        """
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, loader: Any) -> Dict[str, float]:
        """Evaluates model on dataloader.

        Args:
            loader: Validation or test dataloader.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        self.model.eval()
        losses = []
        total_top1 = 0.0
        total_top5 = 0.0
        total_samples = 0
        input_size = None

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                acc = accuracy(outputs, targets, topk=(1, 5))

                batch_size = targets.size(0)
                losses.extend([float(loss.item())] * batch_size)
                total_top1 += (acc["top1"] / 100.0) * batch_size
                total_top5 += (acc["top5"] / 100.0) * batch_size
                total_samples += batch_size

                if input_size is None:
                    input_size = (1, inputs.shape[1], inputs.shape[2], inputs.shape[3])

        avg_loss = average_loss(losses)
        top1 = (total_top1 / max(1, total_samples)) * 100.0
        top5 = (total_top5 / max(1, total_samples)) * 100.0
        params = parameter_count(self.model)
        flops = flops_estimate(self.model, input_size=input_size or (1, 3, 32, 32))
        latency = inference_time(
            model=self.model,
            input_size=input_size or (1, 3, 32, 32),
            device=self.device,
            n_runs=20,
        )

        return {
            "loss": float(avg_loss),
            "accuracy": float(top1),
            "top5_accuracy": float(top5),
            "num_params": int(params),
            "flops": int(flops),
            "inference_time_ms": float(latency),
        }
