"""Memory module for storing architecture trials and outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class ExperimentMemory:
    """Stores architecture attempts and their metrics for RA-NAS.

    The memory is the bridge between search iterations: it allows the
    reasoning agent to learn from previous outcomes and prioritize promising
    architecture patterns.
    """

    def __init__(self) -> None:
        """Initializes an empty in-memory record list."""
        self._entries: List[Dict[str, Any]] = []

    def add(self, arch: Dict[str, Any], metrics: Dict[str, Any], predicted_accuracy: float | None = None) -> None:
        """Adds a single architecture evaluation record.

        Args:
            arch: Architecture dictionary used for model construction.
            metrics: Evaluation metrics dictionary for that architecture.
            predicted_accuracy: LLM's accuracy prediction before training (optional).
        """
        entry: Dict[str, Any] = {"arch": arch, "metrics": metrics}
        if predicted_accuracy is not None:
            entry["predicted_accuracy"] = float(predicted_accuracy)
            entry["prediction_error"] = abs(float(predicted_accuracy) - float(metrics.get("val_accuracy", 0.0)))
        self._entries.append(entry)

    def get_top_k(self, k: int, metric: str = "val_accuracy") -> List[Dict[str, Any]]:
        """Returns top-k records sorted by a target metric in descending order.

        Args:
            k: Number of entries to return.
            metric: Metric key used for sorting.

        Returns:
            List[Dict[str, Any]]: Top entries sorted by metric score.

        Raises:
            ValueError: If k is less than 1.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, received {k}.")
        return sorted(
            self._entries,
            key=lambda item: float(item["metrics"].get(metric, float("-inf"))),
            reverse=True,
        )[:k]

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all memory records in insertion order.

        Returns:
            List[Dict[str, Any]]: Full ordered history.
        """
        return list(self._entries)

    def save(self, path: str) -> None:
        """Saves memory entries to disk as JSON.

        Args:
            path: Output file path.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(self._entries, file, indent=2)

    def load(self, path: str) -> None:
        """Loads memory entries from a JSON file, replacing current state.

        Args:
            path: Input JSON file path.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If loaded content is not a list.
        """
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Memory file not found: {in_path}")
        with in_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, list):
            raise ValueError("Memory payload must be a list of records.")
        self._entries = payload

    def summary(self, k: int = 5, metric: str = "val_accuracy") -> List[Dict[str, Any]]:
        """Returns compact records for prompt-context injection.

        Args:
            k: Maximum number of top entries.
            metric: Metric key used for ranking and surfaced in each compact record.

        Returns:
            List[Dict[str, Any]]: Compact list with architecture and metric score.
        """
        compact: List[Dict[str, Any]] = []
        for item in self.get_top_k(k=k, metric=metric):
            compact.append(
                {
                    "arch": item["arch"],
                    metric: float(item["metrics"].get(metric, 0.0)),
                }
            )
        return compact

