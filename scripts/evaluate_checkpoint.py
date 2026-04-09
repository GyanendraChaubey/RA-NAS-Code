#!/usr/bin/env python3
"""Evaluate a saved RA-NAS checkpoint on a held-out test dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import Evaluator
from src.models.model_builder import build_model
from src.utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for checkpoint evaluation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate an RA-NAS checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-config", type=str, default="configs/train.yaml")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_test_loader(config: Dict[str, Any]) -> DataLoader:
    """Builds test dataloader according to dataset configuration.

    Args:
        config: Training config dictionary.

    Returns:
        DataLoader: Test loader.
    """
    dataset_cfg = config["dataset"]
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    if str(dataset_cfg["name"]).lower() != "cifar10":
        raise ValueError("Only cifar10 dataset evaluation is currently supported.")

    data_dir = Path(dataset_cfg["data_dir"])
    try:
        test_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=False,
            download=True,
            transform=transform,
        )
    except Exception:
        test_dataset = datasets.FakeData(
            size=1000,
            image_size=(3, 32, 32),
            num_classes=int(dataset_cfg["num_classes"]),
            transform=transform,
        )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    """Loads a checkpoint and prints evaluation metrics as JSON."""
    load_dotenv()  # loads .env from project root if present
    args = parse_args()
    config = load_config(args.train_config)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "arch_config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain 'arch_config' and 'state_dict' keys.")

    model = build_model(
        arch_config=checkpoint["arch_config"],
        num_classes=int(config["dataset"]["num_classes"]),
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])

    test_loader = build_test_loader(config)
    evaluator = Evaluator(model=model, device=device)
    metrics = evaluator.evaluate(test_loader)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
