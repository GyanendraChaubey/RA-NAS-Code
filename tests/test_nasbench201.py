"""Tests for the NAS-Bench-201 secondary benchmark track building blocks."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from src.nasbench201.generator import NB201ArchitectureGenerator
from src.nasbench201.prompt_builder import NB201PromptBuilder
from src.nasbench201.search_space import NUM_EDGES, OPS, ops_to_arch_str, validate_ops


# ---------------------------------------------------------------------------
# search_space
# ---------------------------------------------------------------------------

def test_ops_to_arch_str_matches_nats_bench_format() -> None:
    ops = ["nor_conv_3x3", "nor_conv_3x3", "avg_pool_3x3", "skip_connect", "nor_conv_3x3", "skip_connect"]
    assert ops_to_arch_str(ops) == (
        "|nor_conv_3x3~0|+"
        "|nor_conv_3x3~0|avg_pool_3x3~1|+"
        "|skip_connect~0|nor_conv_3x3~1|skip_connect~2|"
    )


def test_ops_to_arch_str_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        ops_to_arch_str(["none"] * 5)


def test_validate_ops_rejects_missing_key() -> None:
    with pytest.raises(ValueError):
        validate_ops({})


def test_validate_ops_rejects_unknown_op() -> None:
    with pytest.raises(ValueError):
        validate_ops({"ops": ["not_a_real_op"] * NUM_EDGES})


def test_validate_ops_accepts_valid_arch() -> None:
    assert validate_ops({"ops": [OPS[0]] * NUM_EDGES}) is True


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

def test_sample_random_is_reproducible_with_seed() -> None:
    gen_a = NB201ArchitectureGenerator(seed=7)
    gen_b = NB201ArchitectureGenerator(seed=7)
    assert gen_a.sample_random() == gen_b.sample_random()


def test_sample_random_produces_valid_ops() -> None:
    gen = NB201ArchitectureGenerator(seed=1)
    for _ in range(20):
        arch = gen.sample_random()
        assert len(arch["ops"]) == NUM_EDGES
        assert all(op in OPS for op in arch["ops"])


def test_mutate_preserves_length_and_validity() -> None:
    gen = NB201ArchitectureGenerator(seed=2)
    arch = gen.sample_random()
    mutated = gen.mutate(arch, num_mutations=3)
    assert len(mutated["ops"]) == NUM_EDGES
    gen.validate(mutated)


def test_mutate_does_not_modify_input_arch() -> None:
    gen = NB201ArchitectureGenerator(seed=3)
    arch = gen.sample_random()
    original = list(arch["ops"])
    gen.mutate(arch, num_mutations=2)
    assert arch["ops"] == original


# ---------------------------------------------------------------------------
# prompt_builder
# ---------------------------------------------------------------------------

def test_schema_includes_prediction_field_when_self_correction_enabled() -> None:
    builder = NB201PromptBuilder(top_k=3, self_correction=True)
    assert "predicted_val_accuracy" in builder._output_schema_text()


def test_schema_omits_prediction_field_when_self_correction_disabled() -> None:
    builder = NB201PromptBuilder(top_k=3, self_correction=False)
    assert "predicted_val_accuracy" not in builder._output_schema_text()


def test_proposal_prompt_includes_diversity_note_when_families_given() -> None:
    builder = NB201PromptBuilder(top_k=3)
    prompt = builder.build_proposal_prompt([], explored_families=["family-a"])
    assert "family-a" in prompt
    assert "Diversity required" in prompt


def test_proposal_prompt_omits_diversity_note_when_no_families() -> None:
    builder = NB201PromptBuilder(top_k=3)
    prompt = builder.build_proposal_prompt([], explored_families=[])
    assert "Diversity required" not in prompt


# ---------------------------------------------------------------------------
# benchmark (nats_bench mocked out — no real API/data file needed)
# ---------------------------------------------------------------------------

class _FakeNatsBenchApi:
    def __init__(self) -> None:
        self.queries: list[tuple[str, str, str]] = []

    def get_more_info(self, arch_str: str, dataset: str, hp: str, is_random: bool = True) -> dict[str, Any]:
        self.queries.append(("get_more_info", arch_str, hp))
        return {
            "train-loss": 1.0, "train-accuracy": 90.0,
            "valid-loss": 1.5, "valid-accuracy": 85.0,
            "test-accuracy": 84.0,
        }

    def get_cost_info(self, arch_str: str, dataset: str, hp: str) -> dict[str, float]:
        self.queries.append(("get_cost_info", arch_str, hp))
        return {"params": 1.23, "flops": 45.6, "latency": 0.01}


@pytest.fixture
def fake_nats_bench_module(monkeypatch: pytest.MonkeyPatch) -> _FakeNatsBenchApi:
    fake_api = _FakeNatsBenchApi()
    fake_module = types.SimpleNamespace(create=lambda *args, **kwargs: fake_api)
    monkeypatch.setitem(sys.modules, "nats_bench", fake_module)
    return fake_api


def test_nats_bench_lookup_maps_fields_to_ranked_metrics(fake_nats_bench_module: _FakeNatsBenchApi) -> None:
    from src.nasbench201.benchmark import NATSBenchLookup

    lookup = NATSBenchLookup(api_file=None, dataset="cifar10-valid")
    metrics = lookup.evaluate({"ops": [OPS[0]] * NUM_EDGES}, hp="200")

    assert metrics["val_accuracy"] == 85.0
    assert metrics["test_accuracy"] == 84.0
    assert metrics["train_accuracy"] == 90.0
    assert metrics["num_params"] == 1.23
    assert metrics["flops"] == 45.6
    assert metrics["inference_time_ms"] == pytest.approx(10.0)
    assert metrics["best_epoch"] == 200


def test_nats_bench_lookup_queries_with_requested_hp(fake_nats_bench_module: _FakeNatsBenchApi) -> None:
    from src.nasbench201.benchmark import NATSBenchLookup

    lookup = NATSBenchLookup(api_file=None, dataset="cifar10-valid")
    lookup.evaluate({"ops": [OPS[0]] * NUM_EDGES}, hp="12")

    assert any(hp == "12" for _, _, hp in fake_nats_bench_module.queries)


def test_nats_bench_lookup_raises_helpful_error_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nats_bench", None)
    from src.nasbench201.benchmark import NATSBenchLookup

    with pytest.raises(ImportError, match="nats_bench"):
        NATSBenchLookup(api_file=None)
