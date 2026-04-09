#!/usr/bin/env python3
"""Run RA-NAS experiments with iterative LLM-guided architecture search."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_agent import LLMAgent
from src.agents.memory import ExperimentMemory
from src.evaluation.evaluator import Evaluator
from src.nas.architecture_generator import ArchitectureGenerator
from src.nas.controller import NASController
from src.nas.search_space import SEARCH_SPACE
from src.training.trainer import Trainer
from src.utils.config_loader import load_config, merge_configs, save_config
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI args.
    """
    parser = argparse.ArgumentParser(description="Run RA-NAS experiment.")
    parser.add_argument("--train-config", type=str, default="configs/train.yaml")
    parser.add_argument("--agent-config", type=str, default="configs/agent.yaml")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy, and PyTorch for reproducible experiments.

    Args:
        seed: Global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(config: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    """Builds train and validation dataloaders from dataset config.

    Args:
        config: Merged config dictionary.
        seed: Seed for deterministic train/val split.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation loaders.
    """
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]

    dataset_name = str(dataset_cfg["name"]).lower()
    if dataset_name != "cifar10":
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Only 'cifar10' is currently implemented.")

    data_dir = Path(dataset_cfg["data_dir"])
    val_split = float(dataset_cfg["val_split"])
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    try:
        full_train_aug = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=True,
            transform=train_transform,
        )
        full_train_eval = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=True,
            transform=eval_transform,
        )
    except Exception:
        # Offline-safe fallback preserves end-to-end behavior in mock mode.
        full_train_aug = datasets.FakeData(
            size=5000,
            image_size=(3, 32, 32),
            num_classes=int(dataset_cfg["num_classes"]),
            transform=train_transform,
        )
        full_train_eval = datasets.FakeData(
            size=5000,
            image_size=(3, 32, 32),
            num_classes=int(dataset_cfg["num_classes"]),
            transform=eval_transform,
        )

    total_size = len(full_train_aug)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("Invalid val_split produced empty train or val split.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_aug, train_indices)
    val_dataset = Subset(full_train_eval, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_summary_table(results: list[Dict[str, Any]]) -> str:
    """Formats final iteration summary table as plain text.

    Args:
        results: Iteration result records.

    Returns:
        str: Multi-line formatted table.
    """
    lines = [
        "iteration | arch summary | val_accuracy | num_params",
        "-" * 90,
    ]
    for record in results:
        arch = record["arch"]
        metrics = record["metrics"]
        arch_summary = (
            f"L={arch['num_layers']} "
            f"F={arch['filters']} "
            f"K={arch['kernels']} "
            f"A={arch['activation']} "
            f"P={arch['pooling']}"
        )
        lines.append(
            f"{record['iteration']:>9} | {arch_summary:<54} | "
            f"{metrics['val_accuracy']:>11.4f} | {int(metrics['num_params']):>10}"
        )
    return "\n".join(lines)


def main() -> None:
    """Executes the full RA-NAS experiment pipeline."""
    load_dotenv()  # loads .env from project root if present
    args = parse_args()
    train_config = load_config(args.train_config)
    agent_config = load_config(args.agent_config)
    merged_config = merge_configs(train_config, agent_config)

    seed = int(merged_config["training"]["seed"])
    seed_everything(seed)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    experiment_cfg = merged_config["experiment"]
    output_dir = Path(experiment_cfg["output_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{experiment_cfg['name']}_{timestamp}"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(name="ra_nas", log_dir=str(experiment_dir))
    logger.info("Starting experiment at %s", experiment_dir)
    logger.info("Device: %s", device)

    save_config(merged_config, str(experiment_dir / "config.yaml"))

    train_loader, val_loader = build_dataloaders(merged_config, seed=seed)
    constraints = merged_config["architecture_constraints"]

    generator = ArchitectureGenerator(
        search_space=SEARCH_SPACE,
        constraints=constraints,
        seed=seed,
    )
    memory = ExperimentMemory()
    agent = LLMAgent(agent_config=merged_config, search_space=generator, memory=memory)

    def trainer_factory(model: torch.nn.Module, iteration_dir: Path) -> Trainer:
        return Trainer(
            model=model,
            config=merged_config,
            device=device,
            experiment_dir=iteration_dir,
            logger=logger,
        )

    def evaluator_factory(model: torch.nn.Module) -> Evaluator:
        return Evaluator(model=model, device=device)

    controller = NASController(
        agent=agent,
        generator=generator,
        trainer=trainer_factory,
        evaluator=evaluator_factory,
        memory=memory,
        config=merged_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        experiment_dir=experiment_dir,
        num_classes=int(merged_config["dataset"]["num_classes"]),
    )

    num_iterations = int(
        args.iterations
        if args.iterations is not None
        else merged_config["agent"]["max_iterations"]
    )
    results = controller.run(num_iterations=num_iterations)

    memory.save(str(experiment_dir / "memory.json"))
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    summary = build_summary_table(results)
    print(summary)
    logger.info("Experiment complete. Results saved to %s", experiment_dir)


if __name__ == "__main__":
    main()
