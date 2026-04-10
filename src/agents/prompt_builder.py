"""Prompt construction utilities for LLM-guided NAS reasoning."""

from __future__ import annotations

import json
from typing import Any, Dict, List


class PromptBuilder:
    """Builds structured prompts for proposing and refining architectures.

    PromptBuilder standardizes prompt formatting so the agent can be swapped
    across LLM providers while preserving RA-NAS prompt semantics.
    """

    def __init__(self, search_space: Dict[str, Any], top_k: int) -> None:
        """Initializes prompt builder.

        Args:
            search_space: Search-space dictionary including constraints.
            top_k: Number of memory entries to include in prompts.
        """
        self.search_space = search_space
        self.top_k = top_k

    def _schema_text(self) -> str:
        """Returns the architecture-only JSON schema including all search dimensions."""
        ss = self.search_space.get("search_space", {})
        activation_opts = "|".join(ss.get("activations", ["relu", "gelu", "silu"]))
        pooling_opts = "|".join(ss.get("pooling", ["max", "avg"]))
        return (
            "{\n"
            '  "num_layers": int,\n'
            '  "filters": [int, ...],\n'
            '  "kernels": [int, ...],\n'
            f'  "activation": "{activation_opts}",\n'
            '  "use_batchnorm": bool,\n'
            '  "use_dropout": bool,\n'
            '  "dropout_rate": float,\n'
            '  "use_skip_connections": bool,\n'
            '  "use_se_blocks": bool,\n'
            f'  "pooling": "{pooling_opts}"\n'
            "}"
        )

    def _output_schema_text(self) -> str:
        """Returns the full response schema including structured reasoning wrapper."""
        return (
            "{\n"
            '  "reasoning": {\n'
            '    "observations": "What patterns do the prior results reveal?",\n'
            '    "hypothesis": "Why should this architecture perform better?",\n'
            '    "changes": "What specific changes are being made and why?",\n'
            '    "risks": "What challenges might this architecture face and how are they mitigated?"\n'
            '  },\n'
            '  "predicted_val_accuracy": float,\n'
            '  "architecture": ' + self._schema_text() + "\n"
            "}"
        )

    def _schema_text(self) -> str:
        """Returns the architecture-only JSON schema."""
        ss = self.search_space.get("search_space", {})
        activation_opts = "|".join(ss.get("activations", ["relu", "gelu", "silu"]))
        pooling_opts = "|".join(ss.get("pooling", ["max", "avg"]))
        return (
            "{\n"
            '  "num_layers": int,\n'
            '  "filters": [int, ...],\n'
            '  "kernels": [int, ...],\n'
            f'  "activation": "{activation_opts}",\n'
            '  "use_batchnorm": bool,\n'
            '  "use_dropout": bool,\n'
            '  "dropout_rate": float,\n'
            '  "use_skip_connections": bool,\n'
            '  "use_se_blocks": bool,\n'
            f'  "pooling": "{pooling_opts}"\n'
            "}"
        )

    def build_proposal_prompt(self, memory_summary: List[Dict[str, Any]], explored_families: List[str] | None = None) -> str:
        """Builds a prompt requesting a fresh architecture proposal."""
        payload = memory_summary[: self.top_k]
        diversity_note = ""
        if explored_families:
            diversity_note = (
                "\nIMPORTANT — Diversity required: the following architecture families have already been "
                "heavily explored. You MUST propose something structurally different "
                "(different num_layers, activation, pooling, use_skip_connections, or use_se_blocks):\n"
                + "\n".join(f"  - {f}" for f in explored_families)
                + "\n"
            )
        return (
            "You are an NAS reasoning agent.\n"
            "Propose a CNN architecture that is valid and likely to improve validation accuracy.\n"
            "Key insight: skip connections (residual additions between layers) and SE blocks "
            "(channel attention) consistently improve accuracy in modern CNNs — prefer "
            "use_skip_connections=true and use_se_blocks=true unless you have a specific reason not to.\n"
            + diversity_note
            + "\nRequired output format (respond ONLY with this JSON, no text outside it):\n"
            f"{self._output_schema_text()}\n\n"
            "Search space and constraints:\n"
            f"{json.dumps(self.search_space, indent=2)}\n\n"
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
        """Builds a prompt requesting an improved variant of an architecture."""
        payload = memory_summary[: self.top_k]
        diversity_note = ""
        if explored_families:
            diversity_note = (
                "\nIMPORTANT — Diversity required: the following families are over-explored. "
                "Try a structurally different architecture:\n"
                + "\n".join(f"  - {f}" for f in explored_families)
                + "\n"
            )
        return (
            "You are an NAS reasoning agent.\n"
            "Refine the given architecture to improve validation performance while staying valid.\n"
            "Key insight: skip connections and SE blocks are high-value — if the current "
            "architecture does not use them, consider enabling them as your primary change.\n"
            + diversity_note
            + "\nRequired output format (respond ONLY with this JSON, no text outside it):\n"
            f"{self._output_schema_text()}\n\n"
            "Search space and constraints:\n"
            f"{json.dumps(self.search_space, indent=2)}\n\n"
            "Architecture to refine:\n"
            f"{json.dumps(arch, indent=2)}\n\n"
            "Performance feedback:\n"
            f"{json.dumps(metrics, indent=2)}\n\n"
            "Top-k prior results (architecture, val_accuracy):\n"
            f"{json.dumps(payload, indent=2)}\n"
        )

