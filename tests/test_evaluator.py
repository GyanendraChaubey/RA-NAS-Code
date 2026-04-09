"""Tests for Evaluator metric correctness."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import Evaluator
from src.models.model_builder import build_model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _simple_arch(num_layers: int = 2) -> dict:
    return {
        "num_layers": num_layers,
        "filters": [16] * num_layers,
        "kernels": [3] * num_layers,
        "activation": "relu",
        "use_batchnorm": False,
        "use_dropout": False,
        "dropout_rate": 0.0,
        "use_skip_connections": False,
        "pooling": "max",
    }


def _make_loader(num_samples: int, num_classes: int, seed: int = 0) -> DataLoader:
    torch.manual_seed(seed)
    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=False)


# ---------------------------------------------------------------------------
# output structure
# ---------------------------------------------------------------------------

def test_evaluate_returns_required_keys() -> None:
    """evaluate() must return all six expected metric keys."""
    model = build_model(_simple_arch(), num_classes=10, device="cpu")
    loader = _make_loader(64, num_classes=10)
    metrics = Evaluator(model, device="cpu").evaluate(loader)
    assert set(metrics.keys()) == {
        "loss", "accuracy", "top5_accuracy", "num_params", "flops", "inference_time_ms"
    }


def test_evaluate_types() -> None:
    """Metric values must have correct Python types."""
    model = build_model(_simple_arch(), num_classes=10, device="cpu")
    loader = _make_loader(64, num_classes=10)
    metrics = Evaluator(model, device="cpu").evaluate(loader)
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["top5_accuracy"], float)
    assert isinstance(metrics["num_params"], int)
    assert isinstance(metrics["flops"], int)
    assert isinstance(metrics["inference_time_ms"], float)


# ---------------------------------------------------------------------------
# accuracy range
# ---------------------------------------------------------------------------

def test_accuracy_range() -> None:
    """Top-1 and Top-5 accuracy must be in [0, 100]."""
    model = build_model(_simple_arch(), num_classes=10, device="cpu")
    loader = _make_loader(128, num_classes=10)
    metrics = Evaluator(model, device="cpu").evaluate(loader)
    assert 0.0 <= metrics["accuracy"] <= 100.0
    assert 0.0 <= metrics["top5_accuracy"] <= 100.0


def test_top5_gte_top1() -> None:
    """Top-5 accuracy must always be >= top-1 accuracy."""
    model = build_model(_simple_arch(), num_classes=10, device="cpu")
    loader = _make_loader(128, num_classes=10)
    metrics = Evaluator(model, device="cpu").evaluate(loader)
    assert metrics["top5_accuracy"] >= metrics["accuracy"]


def test_perfect_accuracy_on_trivial_dataset() -> None:
    """A model whose logit for class 0 is always highest should score 100% on a class-0 dataset."""
    import torch.nn as nn

    class AlwaysClass0(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Dummy parameter so flops_estimate can call next(model.parameters())
            self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            logits = torch.zeros(batch, 10)
            logits[:, 0] = 1e6  # class 0 always wins
            return logits

    model = AlwaysClass0()
    x = torch.randn(32, 3, 32, 32)
    y = torch.zeros(32, dtype=torch.long)  # all class 0
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    metrics = Evaluator(model, device="cpu").evaluate(loader)
    assert metrics["accuracy"] == pytest.approx(100.0, abs=1e-4)
    assert metrics["top5_accuracy"] == pytest.approx(100.0, abs=1e-4)


def test_zero_accuracy_on_worst_case() -> None:
    """A model always predicting class 0 on class-1-only data should score 0% top-1."""
    import torch.nn as nn

    class AlwaysClass0(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            logits = torch.zeros(batch, 10)
            logits[:, 0] = 1e6
            return logits

    model = AlwaysClass0()
    x = torch.randn(32, 3, 32, 32)
    y = torch.ones(32, dtype=torch.long)  # all class 1
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    metrics = Evaluator(model, device="cpu").evaluate(loader)
    assert metrics["accuracy"] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# positive-value sanity checks
# ---------------------------------------------------------------------------

def test_loss_is_positive() -> None:
    """Cross-entropy loss must be > 0 for a randomly initialised model."""
    model = build_model(_simple_arch(), num_classes=10, device="cpu")
    loader = _make_loader(64, num_classes=10)
    metrics = Evaluator(model, device="cpu").evaluate(loader)
    assert metrics["loss"] > 0.0


def test_num_params_positive() -> None:
    """Built model must have trainable parameters."""
    model = build_model(_simple_arch(), num_classes=10, device="cpu")
    loader = _make_loader(32, num_classes=10)
    metrics = Evaluator(model, device="cpu").evaluate(loader)
    assert metrics["num_params"] > 0


def test_flops_positive() -> None:
    """FLOPs estimate must be > 0."""
    model = build_model(_simple_arch(), num_classes=10, device="cpu")
    loader = _make_loader(32, num_classes=10)
    metrics = Evaluator(model, device="cpu").evaluate(loader)
    assert metrics["flops"] > 0


def test_inference_time_positive() -> None:
    """Measured latency must be > 0 ms."""
    model = build_model(_simple_arch(), num_classes=10, device="cpu")
    loader = _make_loader(32, num_classes=10)
    metrics = Evaluator(model, device="cpu").evaluate(loader)
    assert metrics["inference_time_ms"] > 0.0


# ---------------------------------------------------------------------------
# deeper model produces more parameters and flops
# ---------------------------------------------------------------------------

def test_deeper_model_has_more_params_and_flops() -> None:
    """A 4-layer model must have more params and flops than a 2-layer model."""
    loader = _make_loader(32, num_classes=10)
    model_shallow = build_model(_simple_arch(num_layers=2), num_classes=10, device="cpu")
    model_deep = build_model(_simple_arch(num_layers=4), num_classes=10, device="cpu")
    m_shallow = Evaluator(model_shallow, "cpu").evaluate(loader)
    m_deep = Evaluator(model_deep, "cpu").evaluate(loader)
    assert m_deep["num_params"] > m_shallow["num_params"]
    assert m_deep["flops"] > m_shallow["flops"]
