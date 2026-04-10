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
        return (
            "{\n"
            '  "num_layers": int,          // number of stages (2–8)\n'
            '  "filters": [int, ...],      // output channels per stage, e.g. [64,128,256]\n'
            '  "kernels": [int, ...],      // kernel size per stage: 3 or 5\n'
            '  "block_depths": [int, ...], // bottleneck blocks per stage: 1, 2, or 3\n'
            f'  "activation": "{activation_opts}",\n'
            '  "use_batchnorm": true,      // always true\n'
            '  "use_dropout": bool,\n'
            '  "dropout_rate": float,      // 0.0–0.3, 0.0 when use_dropout=false\n'
            '  "use_skip_connections": true, // always true (bottleneck design)\n'
            '  "use_se_blocks": bool,\n'
            '  "pooling": "avg"            // always avg\n'
            "}"
        )

    def _output_schema_text(self) -> str:
        """Returns the full response schema including structured reasoning wrapper."""
        return (
            "{\n"
            '  "reasoning": {\n'
            '    "observations": "What patterns do the prior results reveal?",\n'
            '    "hypothesis": "Why should this architecture perform better?",\n'
            '    "changes": "What specific changes are being made and why (filters, block_depths, kernels, SE)?",\n'
            '    "risks": "What challenges might this architecture face and how are they mitigated?"\n'
            '  },\n'
            '  "predicted_val_accuracy": float,\n'
            '  "architecture": ' + self._schema_text() + "\n"
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
            "You are an NAS reasoning agent searching for high-accuracy ResNet-style CNNs on CIFAR-10.\n"
            "The architecture uses pre-activation ResNet bottleneck blocks. Key searchable dimensions:\n"
            "  - num_layers: number of stages (each stage = stack of bottleneck blocks)\n"
            "  - filters: output channels per stage (64/128/256/512)\n"
            "  - block_depths: how many bottleneck blocks per stage (1/2/3)\n"
            "  - kernels: 3x3 or 5x5 per stage\n"
            "  - use_se_blocks: channel-attention — prefer True for higher accuracy\n"
            "  - use_dropout: light dropout (0.1–0.2) helps regularisation\n"
            "Architecture template for 93%+ accuracy: 4 stages, filters=[64,128,256,512], "
            "block_depths=[2,2,2,2], kernels=[3,3,3,3], use_se_blocks=True.\n"
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
            "You are an NAS reasoning agent searching for high-accuracy ResNet-style CNNs on CIFAR-10.\n"
            "Refine the given architecture to improve validation performance.\n"
            "Focus on: increasing block_depths for more capacity, wider filters (256/512), "
            "SE blocks for channel attention, and 3x3 kernels (most efficient).\n"
            "If val_accuracy is already high (>90%), focus on block_depths and use_se_blocks.\n"
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

