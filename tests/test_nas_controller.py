"""End-to-end tests for NASController iteration loop."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.agents.llm_agent import LLMAgent
from src.agents.memory import ExperimentMemory
from src.evaluation.evaluator import Evaluator
from src.nas.architecture_generator import ArchitectureGenerator
from src.nas.controller import NASController
from src.nas.search_space import SEARCH_SPACE
from src.training.trainer import Trainer
from src.utils.logger import get_logger


# ---------------------------------------------------------------------------
# shared fixtures / helpers
# ---------------------------------------------------------------------------

_CONSTRAINTS = {
    "min_layers": 2,
    "max_layers": 3,
    "min_filters": 64,
    "max_filters": 128,
    "allowed_activations": ["relu"],
    "allowed_kernels": [3],
}

_TRAIN_CFG = {
    "training": {
        "epochs": 1,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "optimizer": "sgd",
        "scheduler": "none",
        "warmup_epochs": 0,
        "augmentation": {
            "cutout": False,
            "mixup": False,
            "randaugment": False,
        },
        "swa": {"enabled": False},
    },
    "early_stopping": {
        "enabled": False,
        "patience": 3,
        "monitor": "val_accuracy",
        "mode": "max",
    },
    "experiment": {"save_best_only": False},
}


def _make_loader(num_samples: int = 64, num_classes: int = 10, seed: int = 0) -> DataLoader:
    torch.manual_seed(seed)
    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=False)


def _build_controller(
    tmp_path: Path,
    num_classes: int = 10,
    explore_every: int = 0,
) -> NASController:
    logger = get_logger("test_controller", str(tmp_path))
    loader = _make_loader(num_classes=num_classes)
    generator = ArchitectureGenerator(SEARCH_SPACE, _CONSTRAINTS, seed=0)
    memory = ExperimentMemory()
    agent = LLMAgent(
        agent_config={
            "llm": {},
            "agent": {
                "mock_mode": True,
                "top_k_memory": 3,
                "retry_on_invalid": 3,
                "feedback_strategy": "top_k",
            },
        },
        search_space=generator,
        memory=memory,
    )
    return NASController(
        agent=agent,
        generator=generator,
        trainer=lambda m, d: Trainer(m, _TRAIN_CFG, "cpu", d, logger),
        evaluator=lambda m: Evaluator(m, "cpu"),
        memory=memory,
        config=_TRAIN_CFG,
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        logger=logger,
        experiment_dir=tmp_path,
        num_classes=num_classes,
        explore_every=explore_every,
    )


# ---------------------------------------------------------------------------
# result structure
# ---------------------------------------------------------------------------

def test_run_returns_correct_number_of_records(tmp_path: Path) -> None:
    """run(n) must return exactly n iteration records."""
    ctrl = _build_controller(tmp_path)
    results = ctrl.run(num_iterations=2)
    assert len(results) == 2


def test_run_record_keys(tmp_path: Path) -> None:
    """Each record must contain 'iteration', 'arch', and 'metrics' keys."""
    ctrl = _build_controller(tmp_path)
    results = ctrl.run(num_iterations=1)
    record = results[0]
    assert set(record.keys()) == {"iteration", "arch", "metrics"}


def test_run_iteration_indices_are_sequential(tmp_path: Path) -> None:
    """Iteration indices must start at 1 and be consecutive."""
    ctrl = _build_controller(tmp_path)
    results = ctrl.run(num_iterations=3)
    assert [r["iteration"] for r in results] == [1, 2, 3]


def test_run_metrics_contain_expected_keys(tmp_path: Path) -> None:
    """Combined metrics dict must contain all nine expected keys."""
    ctrl = _build_controller(tmp_path)
    results = ctrl.run(num_iterations=1)
    metrics = results[0]["metrics"]
    expected = {
        "train_loss", "train_accuracy", "val_loss", "val_accuracy",
        "top5_accuracy", "num_params", "flops", "inference_time_ms", "best_epoch",
    }
    assert expected.issubset(set(metrics.keys()))


def test_run_arches_are_valid(tmp_path: Path) -> None:
    """Every architecture stored in results must pass validation."""
    from src.nas.search_space import validate_architecture
    ctrl = _build_controller(tmp_path)
    results = ctrl.run(num_iterations=2)
    for record in results:
        assert validate_architecture(record["arch"], _CONSTRAINTS) is True


# ---------------------------------------------------------------------------
# memory persistence
# ---------------------------------------------------------------------------

def test_memory_saved_to_disk_after_each_iteration(tmp_path: Path) -> None:
    """memory.json must exist and grow by one entry after each iteration."""
    ctrl = _build_controller(tmp_path)
    memory_path = tmp_path / "memory.json"

    for n in range(1, 4):
        ctrl.run(num_iterations=1)
        assert memory_path.exists(), f"memory.json missing after iteration {n}"
        saved = json.loads(memory_path.read_text())
        assert len(saved) == n, f"expected {n} records, got {len(saved)}"


def test_memory_json_is_valid_json(tmp_path: Path) -> None:
    """memory.json written by the controller must be valid parseable JSON."""
    ctrl = _build_controller(tmp_path)
    ctrl.run(num_iterations=2)
    memory_path = tmp_path / "memory.json"
    data = json.loads(memory_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 2


# ---------------------------------------------------------------------------
# iteration directories
# ---------------------------------------------------------------------------

def test_iteration_directories_created(tmp_path: Path) -> None:
    """iter_001 and iter_002 directories must be created under experiment_dir."""
    ctrl = _build_controller(tmp_path)
    ctrl.run(num_iterations=2)
    assert (tmp_path / "iter_001").is_dir()
    assert (tmp_path / "iter_002").is_dir()


def test_checkpoint_saved_in_iteration_dir(tmp_path: Path) -> None:
    """A model.pt checkpoint must be saved inside each iteration directory."""
    ctrl = _build_controller(tmp_path)
    ctrl.run(num_iterations=1)
    assert (tmp_path / "iter_001" / "model.pt").exists()


# ---------------------------------------------------------------------------
# exploration behaviour
# ---------------------------------------------------------------------------

def test_exploration_triggers_fresh_propose(tmp_path: Path) -> None:
    """With explore_every=2, propose_architecture must be called on iterations 1 and 2."""
    generator = ArchitectureGenerator(SEARCH_SPACE, _CONSTRAINTS, seed=0)
    memory = ExperimentMemory()
    agent = LLMAgent(
        agent_config={
            "llm": {},
            "agent": {
                "mock_mode": True,
                "top_k_memory": 3,
                "retry_on_invalid": 3,
                "feedback_strategy": "top_k",
            },
        },
        search_space=generator,
        memory=memory,
    )

    propose_calls: list = []
    original_propose = agent.propose_architecture

    def tracked_propose():
        arch = original_propose()
        propose_calls.append(arch)
        return arch

    agent.propose_architecture = tracked_propose
    loader = _make_loader()
    logger = get_logger("test_explore", str(tmp_path))

    ctrl = NASController(
        agent=agent,
        generator=generator,
        trainer=lambda m, d: Trainer(m, _TRAIN_CFG, "cpu", d, logger),
        evaluator=lambda m: Evaluator(m, "cpu"),
        memory=memory,
        config=_TRAIN_CFG,
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        logger=logger,
        experiment_dir=tmp_path,
        num_classes=10,
        explore_every=2,
    )
    ctrl.run(num_iterations=4)
    # iterations 1 (init), 2 (2%2==0), 4 (4%2==0) → 3 propose calls
    assert len(propose_calls) == 3


def test_no_exploration_when_disabled(tmp_path: Path) -> None:
    """With explore_every=0, propose_architecture is only called once (iteration 1)."""
    generator = ArchitectureGenerator(SEARCH_SPACE, _CONSTRAINTS, seed=0)
    memory = ExperimentMemory()
    agent = LLMAgent(
        agent_config={
            "llm": {},
            "agent": {
                "mock_mode": True,
                "top_k_memory": 3,
                "retry_on_invalid": 3,
                "feedback_strategy": "top_k",
            },
        },
        search_space=generator,
        memory=memory,
    )

    propose_calls: list = []
    original_propose = agent.propose_architecture

    def tracked_propose():
        arch = original_propose()
        propose_calls.append(arch)
        return arch

    agent.propose_architecture = tracked_propose
    loader = _make_loader()
    logger = get_logger("test_no_explore", str(tmp_path))

    ctrl = NASController(
        agent=agent,
        generator=generator,
        trainer=lambda m, d: Trainer(m, _TRAIN_CFG, "cpu", d, logger),
        evaluator=lambda m: Evaluator(m, "cpu"),
        memory=memory,
        config=_TRAIN_CFG,
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        logger=logger,
        experiment_dir=tmp_path,
        num_classes=10,
        explore_every=0,
    )
    ctrl.run(num_iterations=3)
    assert len(propose_calls) == 1
