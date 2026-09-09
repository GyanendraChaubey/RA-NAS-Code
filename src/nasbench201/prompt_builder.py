"""Prompt construction for LLM-guided search over the NAS-Bench-201 cell space."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from src.nasbench201.search_space import OPS


class NB201PromptBuilder:
    """Builds structured prompts for proposing/refining NAS-Bench-201 genotypes.

    Mirrors src.agents.prompt_builder.PromptBuilder's public interface
    (build_proposal_prompt/build_refinement_prompt) so LLMAgent can use either
    interchangeably via its `prompt_builder` constructor argument.
    """

    def __init__(self, top_k: int, self_correction: bool = True) -> None:
        """Initializes the NAS-Bench-201 prompt builder.

        Args:
            top_k: Number of memory entries to include in prompts.
            self_correction: If True, ask the LLM for a predicted val_accuracy
                so prediction error can be fed back on refinement (Phase 5).
        """
        self.top_k = top_k
        self.self_correction = self_correction

    def _output_schema_text(self) -> str:
        """Returns the full response schema including structured reasoning wrapper."""
        prediction_field = '  "predicted_val_accuracy": float,\n' if self.self_correction else ""
        return (
            "{\n"
            '  "reasoning": {\n'
            '    "observations": "What patterns do the prior results reveal?",\n'
            '    "hypothesis": "Why should this cell perform better?",\n'
            '    "changes": "Which edges are changing operation, and why?",\n'
            '    "risks": "What challenges might this cell face and how are they mitigated?"\n'
            '  },\n'
            + prediction_field
            + '  "architecture": {"ops": [op_1, op_2, op_3, op_4, op_5, op_6]}\n'
            "}"
        )

    def _cell_explainer(self) -> str:
        """Returns the fixed description of the NAS-Bench-201 cell topology."""
        return (
            "You are an NAS reasoning agent searching the NAS-Bench-201 topology cell space "
            "on CIFAR-10. A cell has 4 nodes (0=input, 3=output) and 6 directed edges: "
            "edge1=(0->1), edge2=(0->2), edge3=(1->2), edge4=(0->3), edge5=(1->3), edge6=(2->3). "
            f"Each edge (op_1..op_6) must be one of: {OPS}.\n"
        )

    def build_proposal_prompt(
        self, memory_summary: List[Dict[str, Any]], explored_families: List[str] | None = None
    ) -> str:
        """Builds a prompt requesting a fresh cell proposal."""
        payload = memory_summary[: self.top_k]
        diversity_note = ""
        if explored_families:
            diversity_note = (
                "\nIMPORTANT — Diversity required: the following op combinations have already "
                "been heavily explored. Propose a structurally different combination:\n"
                + "\n".join(f"  - {f}" for f in explored_families)
                + "\n"
            )
        return (
            self._cell_explainer()
            + diversity_note
            + "\nRequired output format (respond ONLY with this JSON, no text outside it):\n"
            f"{self._output_schema_text()}\n\n"
            "Top-k prior results (architecture, val_accuracy):\n"
            f"{json.dumps(payload, indent=2)}\n"
        )

    def build_refinement_prompt(
        self,
        arch: Dict[str, Any],
        metrics: Dict[str, Any],
        memory_summary: List[Dict[str, Any]],
        explored_families: List[str] | None = None,
    ) -> str:
        """Builds a prompt requesting an improved variant of a cell."""
        payload = memory_summary[: self.top_k]
        diversity_note = ""
        if explored_families:
            diversity_note = (
                "\nIMPORTANT — Diversity required: try a structurally different op combination "
                "than these over-explored ones:\n"
                + "\n".join(f"  - {f}" for f in explored_families)
                + "\n"
            )
        return (
            self._cell_explainer()
            + "Refine the given cell to improve validation accuracy by changing 1-2 edges.\n"
            + diversity_note
            + "\nRequired output format (respond ONLY with this JSON, no text outside it):\n"
            f"{self._output_schema_text()}\n\n"
            "Cell to refine:\n"
            f"{json.dumps(arch, indent=2)}\n\n"
            "Performance feedback:\n"
            f"{json.dumps(metrics, indent=2)}\n\n"
            "Top-k prior results (architecture, val_accuracy):\n"
            f"{json.dumps(payload, indent=2)}\n"
        )
