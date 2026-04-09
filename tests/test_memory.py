"""Tests for ExperimentMemory persistence and ranking behavior."""

from pathlib import Path

from src.agents.memory import ExperimentMemory


def test_memory_top_k_and_roundtrip(tmp_path: Path) -> None:
    """Verifies top-k ordering and save/load round-trip fidelity."""
    memory = ExperimentMemory()
    for idx in range(10):
        arch = {
            "num_layers": 2,
            "filters": [16, 32],
            "kernels": [3, 3],
            "activation": "relu",
            "use_batchnorm": True,
            "use_dropout": False,
            "dropout_rate": 0.0,
            "use_skip_connections": False,
            "pooling": "max",
        }
        metrics = {"val_accuracy": float(idx), "val_loss": 10.0 - idx}
        memory.add(arch=arch, metrics=metrics)

    top3 = memory.get_top_k(3, metric="val_accuracy")
    assert len(top3) == 3
    assert top3[0]["metrics"]["val_accuracy"] == 9.0
    assert top3[1]["metrics"]["val_accuracy"] == 8.0
    assert top3[2]["metrics"]["val_accuracy"] == 7.0

    out_path = tmp_path / "memory.json"
    memory.save(str(out_path))

    loaded = ExperimentMemory()
    loaded.load(str(out_path))
    assert loaded.get_all() == memory.get_all()

