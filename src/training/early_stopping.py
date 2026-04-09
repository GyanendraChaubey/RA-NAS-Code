"""Early stopping utility for controlling training budget in RA-NAS."""

from __future__ import annotations

from typing import Any, Dict


class EarlyStopping:
    """Patience-based early stopping for monitored validation metrics.

    This component avoids spending unnecessary compute on weak architectures,
    reducing NAS cost while preserving reproducibility.
    """

    def __init__(self, patience: int, mode: str, monitor: str) -> None:
        """Initializes early stopping state.

        Args:
            patience: Number of unimproved epochs tolerated.
            mode: Optimization direction, either "max" or "min".
            monitor: Metric key to monitor from metrics dictionary.

        Raises:
            ValueError: If mode is unsupported.
        """
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        self.patience = int(patience)
        self.mode = mode
        self.monitor = monitor
        self.reset()

    def reset(self) -> None:
        """Resets early stopping state for a new training run."""
        self.best_value = float("-inf") if self.mode == "max" else float("inf")
        self.bad_epochs = 0

    def step(self, metrics: Dict[str, Any]) -> bool:
        """Updates early stopping state and checks stop condition.

        Args:
            metrics: Metrics dictionary containing monitored metric.

        Returns:
            bool: True when training should stop.

        Raises:
            KeyError: If monitored metric is missing.
        """
        if self.monitor not in metrics:
            raise KeyError(f"Metric '{self.monitor}' not found in metrics: {list(metrics.keys())}")

        value = float(metrics[self.monitor])
        improved = (value > self.best_value) if self.mode == "max" else (value < self.best_value)
        if improved:
            self.best_value = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

