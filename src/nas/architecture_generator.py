"""Architecture sampling and mutation utilities for RA-NAS."""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List

from src.nas.search_space import SEARCH_SPACE, validate_architecture


class ArchitectureGenerator:
    """Generates and mutates valid architectures inside the search space.

    This class isolates stochastic architecture generation from agent logic,
    making it reusable for random-search baselines and fallback handling.
    """

    def __init__(self, search_space: Dict[str, Any], constraints: Dict[str, Any], seed: int = 42) -> None:
        """Initializes architecture generator.

        Args:
            search_space: Global search-space definition dictionary.
            constraints: Config-level architecture constraints.
            seed: Random seed for reproducible sampling.
        """
        self.search_space = search_space
        self.constraints = constraints
        self._rng = random.Random(seed)

    def _filter_choices(self) -> List[int]:
        """Returns valid filter choices after applying config constraints.

        Returns:
            List[int]: Candidate filter values.
        """
        min_filters = int(self.constraints.get("min_filters", min(SEARCH_SPACE["filters_per_layer"])))
        max_filters = int(self.constraints.get("max_filters", max(SEARCH_SPACE["filters_per_layer"])))
        return [
            value
            for value in self.search_space["filters_per_layer"]
            if min_filters <= value <= max_filters
        ]

    def _kernel_choices(self) -> List[int]:
        """Returns allowed kernel choices.

        Returns:
            List[int]: Candidate kernel sizes.
        """
        allowed = set(self.constraints.get("allowed_kernels", self.search_space["kernel_sizes"]))
        return [value for value in self.search_space["kernel_sizes"] if value in allowed]

    def _activation_choices(self) -> List[str]:
        """Returns allowed activation choices.

        Returns:
            List[str]: Candidate activations.
        """
        allowed = set(self.constraints.get("allowed_activations", self.search_space["activations"]))
        return [value for value in self.search_space["activations"] if value in allowed]

    def sample_random(self) -> Dict[str, Any]:
        """Samples a random valid architecture.

        Returns:
            Dict[str, Any]: Valid architecture dictionary.

        Raises:
            ValueError: If constraints leave no valid options.
        """
        min_layers = int(self.constraints.get("min_layers", self.search_space["num_layers"]["min"]))
        max_layers = int(self.constraints.get("max_layers", self.search_space["num_layers"]["max"]))
        filter_choices = self._filter_choices()
        kernel_choices = self._kernel_choices()
        activation_choices = self._activation_choices()

        if not filter_choices:
            raise ValueError("No valid filter choices after applying constraints.")
        if not kernel_choices:
            raise ValueError("No valid kernel choices after applying constraints.")
        if not activation_choices:
            raise ValueError("No valid activation choices after applying constraints.")

        num_layers = self._rng.randint(min_layers, max_layers)
        use_dropout = self._rng.choice([True, False])
        dropout_rate = 0.0
        if use_dropout:
            dropout_rate = round(
                self._rng.uniform(
                    self.search_space["dropout_rate"]["min"],
                    self.search_space["dropout_rate"]["max"],
                ),
                3,
            )

        arch: Dict[str, Any] = {
            "num_layers": num_layers,
            "filters": [self._rng.choice(filter_choices) for _ in range(num_layers)],
            "kernels": [self._rng.choice(kernel_choices) for _ in range(num_layers)],
            "activation": self._rng.choice(activation_choices),
            "use_batchnorm": self._rng.choice(self.search_space["use_batchnorm"]),
            "use_dropout": use_dropout,
            "dropout_rate": dropout_rate,
            "use_skip_connections": self._rng.choice(self.search_space["use_skip_connections"]),
            "use_se_blocks": self._rng.choice(self.search_space.get("use_se_blocks", [True, False])),
            "pooling": self._rng.choice(self.search_space["pooling"]),
        }
        self.validate(arch)
        return arch

    def validate(self, arch: Dict[str, Any]) -> bool:
        """Validates an architecture using shared validation rules.

        Args:
            arch: Architecture dictionary to validate.

        Returns:
            bool: True if valid.
        """
        return validate_architecture(arch=arch, constraints=self.constraints)

    def mutate(self, arch: Dict[str, Any], num_mutations: int = 1) -> Dict[str, Any]:
        """Mutates selected fields of an architecture while preserving validity.

        Args:
            arch: Base architecture.
            num_mutations: Number of fields to perturb.

        Returns:
            Dict[str, Any]: Mutated valid architecture.
        """
        candidates = [
            "num_layers",
            "filters",
            "kernels",
            "activation",
            "use_batchnorm",
            "use_dropout",
            "use_skip_connections",
            "use_se_blocks",
            "pooling",
        ]
        filter_choices = self._filter_choices()
        kernel_choices = self._kernel_choices()
        activation_choices = self._activation_choices()

        mutated = copy.deepcopy(arch)
        for _ in range(max(1, num_mutations)):
            field = self._rng.choice(candidates)
            if field == "num_layers":
                min_layers = int(self.constraints.get("min_layers", self.search_space["num_layers"]["min"]))
                max_layers = int(self.constraints.get("max_layers", self.search_space["num_layers"]["max"]))
                new_depth = self._rng.randint(min_layers, max_layers)
                mutated["num_layers"] = new_depth
                current_filters = mutated["filters"][:new_depth]
                current_kernels = mutated["kernels"][:new_depth]
                while len(current_filters) < new_depth:
                    current_filters.append(self._rng.choice(filter_choices))
                while len(current_kernels) < new_depth:
                    current_kernels.append(self._rng.choice(kernel_choices))
                mutated["filters"] = current_filters
                mutated["kernels"] = current_kernels
            elif field == "filters":
                idx = self._rng.randrange(mutated["num_layers"])
                mutated["filters"][idx] = self._rng.choice(filter_choices)
            elif field == "kernels":
                idx = self._rng.randrange(mutated["num_layers"])
                mutated["kernels"][idx] = self._rng.choice(kernel_choices)
            elif field == "activation":
                mutated["activation"] = self._rng.choice(activation_choices)
            elif field == "use_batchnorm":
                mutated["use_batchnorm"] = not mutated["use_batchnorm"]
            elif field == "use_dropout":
                mutated["use_dropout"] = not mutated["use_dropout"]
                mutated["dropout_rate"] = (
                    round(
                        self._rng.uniform(
                            self.search_space["dropout_rate"]["min"],
                            self.search_space["dropout_rate"]["max"],
                        ),
                        3,
                    )
                    if mutated["use_dropout"]
                    else 0.0
                )
            elif field == "use_skip_connections":
                mutated["use_skip_connections"] = not mutated["use_skip_connections"]
            elif field == "use_se_blocks":
                mutated["use_se_blocks"] = not mutated.get("use_se_blocks", False)
            elif field == "pooling":
                mutated["pooling"] = self._rng.choice(self.search_space["pooling"])

        self.validate(mutated)
        return mutated

