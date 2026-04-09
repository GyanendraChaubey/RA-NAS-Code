"""Unit tests for architecture validation and constraint enforcement."""

import pytest

from src.nas.search_space import validate_architecture


def _constraints():
    return {
        "min_layers": 2,
        "max_layers": 6,
        "min_filters": 16,
        "max_filters": 256,
        "allowed_activations": ["relu", "gelu", "silu"],
        "allowed_kernels": [3, 5],
    }


def _valid_arch():
    return {
        "num_layers": 3,
        "filters": [32, 64, 128],
        "kernels": [3, 3, 5],
        "activation": "relu",
        "use_batchnorm": True,
        "use_dropout": True,
        "dropout_rate": 0.2,
        "use_skip_connections": True,
        "pooling": "max",
    }


def test_valid_architecture_passes() -> None:
    """Valid architecture should pass validation."""
    assert validate_architecture(_valid_arch(), _constraints()) is True


def test_invalid_architecture_raises() -> None:
    """Invalid architecture should raise descriptive ValueError."""
    bad = _valid_arch()
    bad["kernels"] = [7, 3, 5]
    with pytest.raises(ValueError):
        validate_architecture(bad, _constraints())


def test_constraint_filter_range_enforced() -> None:
    """Constraint bounds should reject out-of-range filter values."""
    bad = _valid_arch()
    bad["filters"] = [8, 64, 128]
    with pytest.raises(ValueError):
        validate_architecture(bad, _constraints())

