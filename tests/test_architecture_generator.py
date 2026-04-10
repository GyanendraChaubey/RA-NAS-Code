"""Tests for ArchitectureGenerator sampling and mutation behaviour."""

import pytest

from src.nas.architecture_generator import ArchitectureGenerator
from src.nas.search_space import SEARCH_SPACE, validate_architecture


def _make_generator(seed: int = 0) -> ArchitectureGenerator:
    constraints = {
        "min_layers": 2,
        "max_layers": 6,
        "min_filters": 16,
        "max_filters": 256,
        "allowed_activations": ["relu", "gelu", "silu"],
        "allowed_kernels": [3, 5],
    }
    return ArchitectureGenerator(SEARCH_SPACE, constraints, seed=seed)


def _valid_base_arch() -> dict:
    return {
        "num_layers": 3,
        "filters": [32, 64, 128],
        "kernels": [3, 5, 3],
        "activation": "relu",
        "use_batchnorm": True,
        "use_dropout": False,
        "dropout_rate": 0.0,
        "use_skip_connections": False,
        "use_se_blocks": False,
        "pooling": "max",
    }


# ---------------------------------------------------------------------------
# sample_random
# ---------------------------------------------------------------------------

def test_sample_random_is_valid() -> None:
    """Randomly sampled architecture must pass validation."""
    gen = _make_generator()
    arch = gen.sample_random()
    assert validate_architecture(arch, gen.constraints) is True


def test_sample_random_respects_layer_bounds() -> None:
    """Sampled num_layers must stay within configured [min, max]."""
    gen = _make_generator()
    for _ in range(20):
        arch = gen.sample_random()
        assert 2 <= arch["num_layers"] <= 6


def test_sample_random_list_lengths_match_num_layers() -> None:
    """filters and kernels lists must have exactly num_layers entries."""
    gen = _make_generator()
    for _ in range(10):
        arch = gen.sample_random()
        assert len(arch["filters"]) == arch["num_layers"]
        assert len(arch["kernels"]) == arch["num_layers"]


def test_sample_random_is_reproducible() -> None:
    """Same seed must produce identical sequences of architectures."""
    archs_a = [_make_generator(seed=7).sample_random() for _ in range(5)]
    archs_b = [_make_generator(seed=7).sample_random() for _ in range(5)]
    assert archs_a == archs_b


def test_sample_random_different_seeds_differ() -> None:
    """Different seeds should (overwhelmingly) produce different results."""
    arch_a = _make_generator(seed=1).sample_random()
    arch_b = _make_generator(seed=2).sample_random()
    assert arch_a != arch_b


# ---------------------------------------------------------------------------
# mutate — structural correctness
# ---------------------------------------------------------------------------

def test_mutate_returns_valid_architecture() -> None:
    """Mutated architecture must pass full validation."""
    gen = _make_generator()
    for _ in range(30):
        arch = gen.sample_random()
        mutated = gen.mutate(arch, num_mutations=1)
        assert validate_architecture(mutated, gen.constraints) is True


def test_mutate_produces_different_arch() -> None:
    """With enough mutations over many trials, at least one must differ."""
    gen = _make_generator(seed=42)
    base = _valid_base_arch()
    results = [gen.mutate(base, num_mutations=1) for _ in range(50)]
    assert any(r != base for r in results)


def test_mutate_list_lengths_consistent_after_depth_change() -> None:
    """After a num_layers mutation, filters/kernels lengths must match new depth."""
    gen = _make_generator(seed=99)
    for _ in range(50):
        arch = gen.sample_random()
        mutated = gen.mutate(arch, num_mutations=3)
        assert len(mutated["filters"]) == mutated["num_layers"]
        assert len(mutated["kernels"]) == mutated["num_layers"]


def test_mutate_num_mutations_applies_multiple_changes() -> None:
    """num_mutations > 1 should still produce a valid architecture."""
    gen = _make_generator()
    base = gen.sample_random()
    mutated = gen.mutate(base, num_mutations=5)
    assert validate_architecture(mutated, gen.constraints) is True


def test_mutate_dropout_rate_zero_when_disabled() -> None:
    """If use_dropout is toggled off, dropout_rate must be 0.0."""
    gen = _make_generator(seed=0)
    # Force a base arch with dropout enabled so toggling is meaningful.
    base = _valid_base_arch()
    base["use_dropout"] = True
    base["dropout_rate"] = 0.3
    # Run many mutations; any that toggle dropout off must zero the rate.
    for _ in range(100):
        mutated = gen.mutate(base, num_mutations=1)
        if not mutated["use_dropout"]:
            assert mutated["dropout_rate"] == 0.0


# ---------------------------------------------------------------------------
# constraint enforcement
# ---------------------------------------------------------------------------

def test_sample_random_honours_filter_constraints() -> None:
    """Sampled filter values must not exceed configured max_filters."""
    constraints = {
        "min_layers": 2,
        "max_layers": 4,
        "min_filters": 16,
        "max_filters": 32,  # tight upper bound
        "allowed_activations": ["relu"],
        "allowed_kernels": [3],
    }
    gen = ArchitectureGenerator(SEARCH_SPACE, constraints, seed=0)
    for _ in range(20):
        arch = gen.sample_random()
        assert all(f <= 32 for f in arch["filters"])


def test_empty_filter_choices_raises() -> None:
    """Impossible filter constraint should raise ValueError."""
    constraints = {
        "min_filters": 999,
        "max_filters": 1000,
        "min_layers": 2,
        "max_layers": 4,
        "allowed_activations": ["relu"],
        "allowed_kernels": [3],
    }
    gen = ArchitectureGenerator(SEARCH_SPACE, constraints, seed=0)
    with pytest.raises(ValueError, match="filter"):
        gen.sample_random()
