"""LLM-based reasoning agent for proposing and refining NAS architectures."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict

from src.agents.memory import ExperimentMemory
from src.agents.prompt_builder import PromptBuilder
from src.nas.architecture_generator import ArchitectureGenerator


class LLMAgent:
    """Reasoning agent that proposes/refines architectures using an LLM.

    LLMAgent encapsulates all model-provider logic. In production mode it
    calls an OpenAI-compatible API; in mock mode it samples valid architectures
    without external network calls, enabling reproducible offline tests.
    """

    def __init__(
        self,
        agent_config: Dict[str, Any],
        search_space: ArchitectureGenerator,
        memory: ExperimentMemory,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        """Initializes the LLM reasoning agent.

        Args:
            agent_config: Agent and provider configuration dictionary.
            search_space: Architecture generator/validator interface.
            memory: Shared experiment memory for prompt context.
            prompt_builder: Optional pre-built PromptBuilder. If None, one is
                constructed automatically from the generator's search space.

        Raises:
            ValueError: If provider is unsupported in non-mock mode.
        """
        self.agent_config = agent_config
        self.generator = search_space
        self.memory = memory
        self.logger = logging.getLogger(__name__)

        llm_cfg = agent_config.get("llm", {})
        agent_cfg = agent_config.get("agent", {})
        self.mock_mode = bool(agent_cfg.get("mock_mode", False))
        self.retry_on_invalid = int(agent_cfg.get("retry_on_invalid", 3))
        self.temperature = float(llm_cfg.get("temperature", 0.7))
        self.max_tokens = int(llm_cfg.get("max_tokens", 1024))
        self.top_k_memory = int(agent_cfg.get("top_k_memory", 5))
        self.feedback_strategy = str(agent_cfg.get("feedback_strategy", "top_k"))

        if prompt_builder is not None:
            self.prompt_builder = prompt_builder
        else:
            prompt_space = {
                "search_space": self.generator.search_space,
                "constraints": self.generator.constraints,
            }
            self.prompt_builder = PromptBuilder(prompt_space, top_k=self.top_k_memory)

        self._client = None
        self._model = str(llm_cfg.get("model", "gpt-4o"))
        provider = str(llm_cfg.get("provider", "openai")).lower()
        if not self.mock_mode:
            if provider not in {"openai", "deepseek", "groq"}:
                raise ValueError(
                    f"Unsupported llm.provider='{provider}'. Supported values: openai, deepseek, groq."
                )
            self._init_openai_client(llm_cfg)

    def _init_openai_client(self, llm_cfg: Dict[str, Any]) -> None:
        """Creates an OpenAI-compatible client from environment-backed configuration.

        Supports any OpenAI-compatible provider (OpenAI, DeepSeek, etc.) via an
        optional ``base_url`` config key.

        Args:
            llm_cfg: LLM config dictionary containing api_key_env and optional base_url.

        Raises:
            ValueError: If API key env var is not set.
            ImportError: If openai package is missing.
        """
        api_key_env = str(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key environment variable '{api_key_env}'. "
                "Set it or enable agent.mock_mode=true."
            )
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("openai package is required for non-mock mode.") from error
        base_url = llm_cfg.get("base_url") or None
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def propose_architecture(self) -> Dict[str, Any]:
        """Proposes a valid architecture from memory and search constraints.

        Returns:
            Dict[str, Any]: Valid architecture dictionary.
        """
        if self.mock_mode:
            return self.generator.sample_random()

        memory_summary = self._select_memory_summary()
        prompt = self.prompt_builder.build_proposal_prompt(memory_summary)
        return self._generate_valid_architecture(prompt)

    def refine_architecture(self, arch: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Refines an architecture using metric feedback and memory context.

        Args:
            arch: Previous architecture to improve.
            feedback: Training/evaluation metrics for the architecture.

        Returns:
            Dict[str, Any]: Improved valid architecture.
        """
        if self.mock_mode:
            return self.generator.mutate(arch, num_mutations=2)

        memory_summary = self._select_memory_summary(reference_metrics=feedback)
        prompt = self.prompt_builder.build_refinement_prompt(arch, feedback, memory_summary)
        return self._generate_valid_architecture(prompt, fallback_arch=arch)

    def _select_memory_summary(
        self,
        reference_metrics: Dict[str, Any] | None = None,
    ) -> list[Dict[str, Any]]:
        """Selects memory context according to configured feedback strategy.

        Args:
            reference_metrics: Optional current metrics for thresholding.

        Returns:
            list[Dict[str, Any]]: Prompt-ready memory summary items.
        """
        strategy = self.feedback_strategy.lower()
        if strategy == "top_k":
            return self.memory.summary(k=self.top_k_memory)
        if strategy == "all":
            return [
                {
                    "arch": entry["arch"],
                    "val_accuracy": float(entry["metrics"].get("val_accuracy", 0.0)),
                }
                for entry in self.memory.get_all()
            ][: self.top_k_memory]
        if strategy == "threshold":
            all_entries = self.memory.get_all()
            if not all_entries:
                return []
            if reference_metrics and "val_accuracy" in reference_metrics:
                threshold = float(reference_metrics["val_accuracy"])
            else:
                values = [float(item["metrics"].get("val_accuracy", 0.0)) for item in all_entries]
                threshold = sum(values) / max(1, len(values))
            selected = [
                {
                    "arch": item["arch"],
                    "val_accuracy": float(item["metrics"].get("val_accuracy", 0.0)),
                }
                for item in all_entries
                if float(item["metrics"].get("val_accuracy", 0.0)) >= threshold
            ]
            if not selected:
                return self.memory.summary(k=self.top_k_memory)
            return selected[: self.top_k_memory]
        raise ValueError(
            "Unsupported agent.feedback_strategy. Use one of: top_k, threshold, all."
        )

    def _generate_valid_architecture(
        self,
        prompt: str,
        fallback_arch: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Calls the LLM and retries parsing/validation before fallback.

        Args:
            prompt: Prompt text to send to the LLM.
            fallback_arch: Optional architecture to mutate for fallback.

        Returns:
            Dict[str, Any]: Valid architecture.
        """
        for attempt in range(1, self.retry_on_invalid + 1):
            response = self._call_llm(prompt)
            try:
                arch = self._parse_response(response)
                self.generator.validate(arch)
                return arch
            except ValueError as error:
                self.logger.warning(
                    "Invalid architecture from LLM at attempt %d/%d: %s",
                    attempt,
                    self.retry_on_invalid,
                    error,
                )

        self.logger.warning(
            "Falling back after %d invalid LLM responses.",
            self.retry_on_invalid,
        )
        if fallback_arch is not None:
            return self.generator.mutate(fallback_arch, num_mutations=2)
        return self.generator.sample_random()

    def _call_llm(self, prompt: str) -> str:
        """Calls the configured LLM API with retry on transient failures.

        Args:
            prompt: Prompt text sent to the model.

        Returns:
            str: Raw text response from the model.

        Raises:
            RuntimeError: If all retries fail.
        """
        if self.mock_mode:
            return json.dumps(self.generator.sample_random())

        if self._client is None:
            raise RuntimeError("LLM client is not initialized.")

        backoff_seconds = [1.0, 2.0, 4.0]
        last_error: Exception | None = None
        for sleep_seconds in backoff_seconds:
            try:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = completion.choices[0].message.content
                if isinstance(content, list):
                    # Defensive handling for structured SDK content variants.
                    text_content = "".join(
                        chunk.get("text", "") for chunk in content if isinstance(chunk, dict)
                    )
                    return text_content
                return str(content or "")
            except Exception as error:  # pylint: disable=broad-except
                last_error = error
                self.logger.warning("LLM call failed: %s. Retrying...", error)
                time.sleep(sleep_seconds)

        raise RuntimeError(f"LLM call failed after retries: {last_error}")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Extracts and parses an architecture JSON object from model text.

        Args:
            response: Raw response text from the LLM.

        Returns:
            Dict[str, Any]: Parsed architecture dictionary.

        Raises:
            ValueError: If no valid JSON object can be extracted.
        """
        cleaned = response.strip()
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

        candidates = [fenced]
        match = re.search(r"\{.*\}", fenced, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise ValueError("LLM response JSON must be an object.")
                return parsed
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not parse architecture JSON from response: {response[:200]}")
