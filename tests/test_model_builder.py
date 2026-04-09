"""Unit tests for model building and shape/size invariants."""

import torch

from src.models.model_builder import build_model, count_parameters


def _arch_configs():
    return [
        {
            "num_layers": 2,
            "filters": [16, 32],
            "kernels": [3, 3],
            "activation": "relu",
            "use_batchnorm": True,
            "use_dropout": False,
            "dropout_rate": 0.0,
            "use_skip_connections": False,
            "pooling": "max",
        },
        {
            "num_layers": 3,
            "filters": [32, 64, 64],
            "kernels": [3, 5, 3],
            "activation": "gelu",
            "use_batchnorm": True,
            "use_dropout": True,
            "dropout_rate": 0.2,
            "use_skip_connections": True,
            "pooling": "avg",
        },
        {
            "num_layers": 4,
            "filters": [16, 32, 64, 128],
            "kernels": [5, 3, 3, 5],
            "activation": "silu",
            "use_batchnorm": False,
            "use_dropout": True,
            "dropout_rate": 0.1,
            "use_skip_connections": True,
            "pooling": "max",
        },
    ]


def test_build_model_output_shape_and_params() -> None:
    """Built models should produce expected output shape and have parameters."""
    batch_size = 4
    num_classes = 10
    x = torch.randn(batch_size, 3, 32, 32)

    for arch in _arch_configs():
        model = build_model(arch_config=arch, num_classes=num_classes, device="cpu")
        y = model(x)
        assert y.shape == (batch_size, num_classes)
        assert count_parameters(model) > 0

