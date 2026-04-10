"""Smoke tests for trainer execution on synthetic data."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.model_builder import build_model
from src.training.trainer import Trainer
from src.utils.logger import get_logger


class EasyBinaryDataset(Dataset):
    """Small separable dataset for quick deterministic training tests."""

    def __init__(self, size: int = 128, seed: int = 42) -> None:
        """Creates synthetic binary classification samples.

        Args:
            size: Number of samples.
            seed: Random seed for deterministic data generation.
        """
        super().__init__()
        torch.manual_seed(seed)
        half = size // 2
        x0 = torch.randn(half, 3, 32, 32) * 0.05
        x1 = torch.ones(size - half, 3, 32, 32) + torch.randn(size - half, 3, 32, 32) * 0.05
        y0 = torch.zeros(half, dtype=torch.long)
        y1 = torch.ones(size - half, dtype=torch.long)
        self.x = torch.cat([x0, x1], dim=0)
        self.y = torch.cat([y0, y1], dim=0)

    def __len__(self) -> int:
        """Returns dataset size."""
        return len(self.y)

    def __getitem__(self, index: int):
        """Returns one sample and label."""
        return self.x[index], self.y[index]


def test_trainer_smoke(tmp_path: Path) -> None:
    """Runs two epochs and checks metric structure and loss trend."""
    arch = {
        "num_layers": 2,
        "filters": [64, 128],
        "kernels": [3, 3],
        "block_depths": [1, 1],
        "activation": "relu",
        "use_batchnorm": True,
        "use_dropout": False,
        "dropout_rate": 0.0,
        "use_skip_connections": True,
        "use_se_blocks": False,
        "pooling": "avg",
    }
    config = {
        "training": {
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "optimizer": "sgd",
            "scheduler": "none",
            "warmup_epochs": 0,
            "seed": 42,
            "augmentation": {
                "cutout": False,
                "mixup": False,
                "randaugment": False,
            },
            "swa": {"enabled": False},
        },
        "early_stopping": {
            "enabled": False,
            "patience": 5,
            "monitor": "val_accuracy",
            "mode": "max",
        },
        "experiment": {
            "save_best_only": False,
        },
    }

    model = build_model(arch_config=arch, num_classes=2, device="cpu")
    logger = get_logger(name="test_trainer", log_dir=str(tmp_path))
    trainer = Trainer(
        model=model,
        config=config,
        device="cpu",
        experiment_dir=tmp_path,
        logger=logger,
    )

    dataset = EasyBinaryDataset(size=128, seed=42)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    metrics = trainer.train(train_loader=train_loader, val_loader=val_loader)
    assert "history" in metrics
    assert len(metrics["history"]) >= 1
    losses = [float(epoch["train_loss"]) for epoch in metrics["history"]]
    assert losses[-1] <= losses[0] + 1e-6

