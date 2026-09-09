"""Tests for LLMAgent self-correction toggle and cost/usage accounting."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.llm_agent import LLMAgent
from src.agents.memory import ExperimentMemory
from src.agents.prompt_builder import PromptBuilder
from src.nas.architecture_generator import ArchitectureGenerator
from src.nas.search_space import SEARCH_SPACE

_CONSTRAINTS = {
    "min_layers": 2,
    "max_layers": 3,
    "min_filters": 64,
    "max_filters": 128,
    "allowed_activations": ["relu"],
    "allowed_kernels": [3],
}

_VALID_ARCH = {
    "num_layers": 2,
    "filters": [64, 128],
    "kernels": [3, 3],
    "block_depths": [2, 2],
    "activation": "relu",
    "use_batchnorm": True,
    "use_dropout": False,
    "dropout_rate": 0.0,
    "use_skip_connections": True,
    "use_se_blocks": False,
    "pooling": "avg",
}


def _response_json(predicted_val_accuracy: float | None) -> str:
    payload: dict[str, Any] = {
        "reasoning": {
            "observations": "obs", "hypothesis": "hyp", "changes": "chg", "risks": "risk",
        },
        "architecture": _VALID_ARCH,
    }
    if predicted_val_accuracy is not None:
        payload["predicted_val_accuracy"] = predicted_val_accuracy
    return json.dumps(payload)


class _FakeClient:
    """Minimal stand-in for the OpenAI client's chat.completions.create surface."""

    def __init__(self, content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> None:
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.call_count = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs: Any) -> SimpleNamespace:
        self.call_count += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens),
        )


class _SequencedFakeClient:
    """Fake client that returns a different response content on each successive call."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.call_count = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs: Any) -> SimpleNamespace:
        content = self.contents[min(self.call_count, len(self.contents) - 1)]
        self.call_count += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


def _build_agent(
    monkeypatch: pytest.MonkeyPatch,
    self_correction: bool,
    pricing: dict[str, float] | None = None,
) -> LLMAgent:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    generator = ArchitectureGenerator(SEARCH_SPACE, _CONSTRAINTS, seed=0)
    memory = ExperimentMemory()
    llm_cfg: dict[str, Any] = {
        "provider": "openai",
        "model": "fake-model",
        "api_key_env": "FAKE_API_KEY",
        "max_tokens": 256,
    }
    if pricing is not None:
        llm_cfg["pricing"] = pricing
    return LLMAgent(
        agent_config={
            "llm": llm_cfg,
            "agent": {
                "mock_mode": False,
                "top_k_memory": 3,
                "retry_on_invalid": 1,
                "feedback_strategy": "top_k",
                "diversity_penalty": False,
                "self_correction": self_correction,
            },
        },
        search_space=generator,
        memory=memory,
    )


# ---------------------------------------------------------------------------
# Prompt schema toggle
# ---------------------------------------------------------------------------

def test_schema_includes_prediction_field_when_self_correction_enabled() -> None:
    builder = PromptBuilder({"search_space": SEARCH_SPACE, "constraints": {}}, top_k=3, self_correction=True)
    assert "predicted_val_accuracy" in builder._output_schema_text()


def test_schema_omits_prediction_field_when_self_correction_disabled() -> None:
    builder = PromptBuilder({"search_space": SEARCH_SPACE, "constraints": {}}, top_k=3, self_correction=False)
    assert "predicted_val_accuracy" not in builder._output_schema_text()


def test_agent_builds_prompt_builder_matching_self_correction_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _build_agent(monkeypatch, self_correction=False)
    assert agent.prompt_builder.self_correction is False


# ---------------------------------------------------------------------------
# Self-correction gating
# ---------------------------------------------------------------------------

def test_prediction_is_captured_when_self_correction_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _build_agent(monkeypatch, self_correction=True)
    agent._client = _FakeClient(_response_json(predicted_val_accuracy=88.0))

    agent.propose_architecture()

    assert agent._last_predicted_accuracy == 88.0


def test_prediction_is_ignored_when_self_correction_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _build_agent(monkeypatch, self_correction=False)
    # LLM includes the field anyway; the agent must not use it when disabled.
    agent._client = _FakeClient(_response_json(predicted_val_accuracy=99.0))

    agent.propose_architecture()

    assert agent._last_predicted_accuracy is None


def test_refine_resets_prediction_before_reusing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-refine prediction must be consumed (reset to None) before the
    refine call's own response is parsed — otherwise a stale prediction from
    the previous architecture could leak into the next one's error tracking.
    """
    agent = _build_agent(monkeypatch, self_correction=True)
    agent._client = _SequencedFakeClient([
        _response_json(predicted_val_accuracy=88.0),
        _response_json(predicted_val_accuracy=None),
    ])

    agent.propose_architecture()
    assert agent._last_predicted_accuracy == 88.0

    agent.refine_architecture(arch=_VALID_ARCH, feedback={"val_accuracy": 80.0})
    assert agent._last_predicted_accuracy is None


# ---------------------------------------------------------------------------
# Cost/usage accounting
# ---------------------------------------------------------------------------

def test_cost_summary_accumulates_tokens_and_estimated_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    pricing = {"input_per_million_usd": 1000.0, "output_per_million_usd": 2000.0}
    agent = _build_agent(monkeypatch, self_correction=True, pricing=pricing)
    agent._client = _FakeClient(_response_json(predicted_val_accuracy=88.0), prompt_tokens=100, completion_tokens=50)

    agent.propose_architecture()
    summary = agent.get_cost_summary()

    assert summary["total_llm_calls"] == 1
    assert summary["total_prompt_tokens"] == 100
    assert summary["total_completion_tokens"] == 50
    assert summary["estimated_cost_usd"] == pytest.approx(0.2)


def test_cost_summary_defaults_to_zero_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _build_agent(monkeypatch, self_correction=True)
    agent._client = _FakeClient(_response_json(predicted_val_accuracy=88.0))

    agent.propose_architecture()
    summary = agent.get_cost_summary()

    assert summary["estimated_cost_usd"] == 0.0
