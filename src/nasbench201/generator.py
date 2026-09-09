"""Random sampling and mutation for the NAS-Bench-201 topology search space."""

from __future__ import annotations

import random
from typing import Any, Dict

from src.nasbench201.search_space import NUM_EDGES, OPS, validate_ops


class NB201ArchitectureGenerator:
    """Samples and mutates NAS-Bench-201 cell genotypes.

    Mirrors src.nas.architecture_generator.ArchitectureGenerator's public
    interface (sample_random/mutate/validate) so it plugs directly into
    LLMAgent without any changes to the agent itself.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initializes the generator.

        Args:
            seed: Random seed for reproducible sampling.
        """
        self._rng = random.Random(seed)
        # Exposed for parity with ArchitectureGenerator; read by NB201PromptBuilder callers.
        self.search_space: Dict[str, Any] = {"ops": OPS, "num_edges": NUM_EDGES}
        self.constraints: Dict[str, Any] = {}

    def sample_random(self) -> Dict[str, Any]:
        """Samples a random valid NAS-Bench-201 genotype.

        Returns:
            Dict[str, Any]: Architecture dict with an "ops" key.
        """
        arch = {"ops": [self._rng.choice(OPS) for _ in range(NUM_EDGES)]}
        self.validate(arch)
        return arch

    def validate(self, arch: Dict[str, Any]) -> bool:
        """Validates a NAS-Bench-201 architecture dict.

        Args:
            arch: Architecture dictionary to validate.

        Returns:
            bool: True if valid.
        """
        return validate_ops(arch)

    def mutate(self, arch: Dict[str, Any], num_mutations: int = 1) -> Dict[str, Any]:
        """Mutates a random subset of edges to different operations.

        Args:
            arch: Base architecture.
            num_mutations: Number of edges to perturb.

        Returns:
            Dict[str, Any]: Mutated valid architecture.
        """
        mutated = {"ops": list(arch["ops"])}
        for _ in range(max(1, num_mutations)):
            idx = self._rng.randrange(NUM_EDGES)
            mutated["ops"][idx] = self._rng.choice(OPS)
        self.validate(mutated)
        return mutated
