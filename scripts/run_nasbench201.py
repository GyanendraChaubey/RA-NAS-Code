#!/usr/bin/env python3
"""Run RA-NAS's LLM reasoning agent against the NAS-Bench-201 topology space.

This is a secondary benchmark track: instead of training candidate CNNs on
CIFAR-10 with the ResNet-bottleneck search space (scripts/run_experiment.py),
architectures are evaluated with an instant lookup against NATS-Bench's
precomputed results. Same LLMAgent, same ablation toggles (diversity_penalty,
self_correction, mock_mode+explore_every for a random-search baseline) — only
the search space and the evaluation backend change. This gives a directly
comparable number against other LLM-NAS papers that report on this benchmark.

Requires the `nats_bench` package and a downloaded NATS-tss benchmark file;
see configs/nasbench201.yaml and the README's "NAS-Bench-201 track" section.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_agent import LLMAgent
from src.agents.memory import ExperimentMemory
from src.nasbench201.benchmark import NATSBenchLookup
from src.nasbench201.generator import NB201ArchitectureGenerator
from src.nasbench201.prompt_builder import NB201PromptBuilder
from src.utils.config_loader import load_config, merge_configs, save_config
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI args.
    """
    parser = argparse.ArgumentParser(description="Run RA-NAS on the NAS-Bench-201 benchmark.")
    parser.add_argument("--agent-config", type=str, default="configs/agent.yaml")
    parser.add_argument("--nasbench-config", type=str, default="configs/nasbench201.yaml")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--api-file", type=str, default=None, help="Overrides nasbench201.api_file.")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Additional YAML files merged on top of --agent-config/--nasbench-config, in "
        "order (later files win). Use for ablations, e.g. configs/ablations/no_diversity_penalty.yaml.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seeds Python and NumPy for reproducible sampling.

    Args:
        seed: Global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def run(
    agent: LLMAgent,
    generator: NB201ArchitectureGenerator,
    benchmark: NATSBenchLookup,
    memory: ExperimentMemory,
    num_iterations: int,
    explore_every: int,
    screening_enabled: bool,
    screen_hp: str,
    full_hp: str,
    logger: Any,
) -> List[Dict[str, Any]]:
    """Runs the propose/refine loop against the NAS-Bench-201 lookup backend.

    Args:
        agent: LLM reasoning agent (any diversity_penalty/self_correction config).
        generator: NAS-Bench-201 architecture generator/validator.
        benchmark: NATS-Bench lookup wrapper.
        memory: Shared experiment memory.
        num_iterations: Number of NAS iterations to run.
        explore_every: Force a fresh proposal every N iterations (0 = disable).
        screening_enabled: If True, use multi-fidelity screening (screen_hp then full_hp).
        screen_hp: Epoch-budget key for screening queries.
        full_hp: Epoch-budget key for full-fidelity queries.
        logger: Experiment logger instance.

    Returns:
        List[Dict[str, Any]]: One record per iteration (arch, metrics, cost).
    """
    results: List[Dict[str, Any]] = []
    screening_buffer: List[Dict[str, Any]] = []
    candidate_arch: Dict[str, Any] | None = None
    explore_every = max(1, explore_every) if explore_every > 0 else 0

    for iteration in range(1, num_iterations + 1):
        if candidate_arch is None or (explore_every > 0 and iteration % explore_every == 0):
            arch = agent.propose_architecture()
        else:
            arch = candidate_arch
        generator.validate(arch)

        if screening_enabled:
            screen_metrics = benchmark.evaluate(arch, hp=screen_hp)
            screening_buffer.append({"arch": arch, "metrics": screen_metrics})
            if len(screening_buffer) >= 2:
                prev, curr = screening_buffer[-2], screening_buffer[-1]
                winner = prev if prev["metrics"]["val_accuracy"] >= curr["metrics"]["val_accuracy"] else curr
                screening_buffer.clear()
                arch = winner["arch"]
                metrics = benchmark.evaluate(arch, hp=full_hp)
            else:
                metrics = screen_metrics
        else:
            metrics = benchmark.evaluate(arch, hp=full_hp)

        memory.add(arch, metrics, predicted_accuracy=getattr(agent, "_last_predicted_accuracy", None))
        candidate_arch = agent.refine_architecture(arch=arch, feedback=metrics)
        generator.validate(candidate_arch)

        cost_summary = agent.get_cost_summary() if hasattr(agent, "get_cost_summary") else {}
        results.append({"iteration": iteration, "arch": arch, "metrics": metrics, "cost": cost_summary})
        logger.info(
            "Iteration=%d | ops=%s | val_acc=%.4f test_acc=%.4f hp=%d",
            iteration, arch["ops"], metrics["val_accuracy"], metrics["test_accuracy"], metrics["best_epoch"],
        )

    return results


def build_summary_table(results: List[Dict[str, Any]]) -> str:
    """Formats final iteration summary table as plain text.

    Args:
        results: Iteration result records.

    Returns:
        str: Multi-line formatted table.
    """
    lines = [
        "iteration | ops                                                        | val_accuracy | test_accuracy",
        "-" * 100,
    ]
    for record in results:
        ops_summary = ",".join(record["arch"]["ops"])
        metrics = record["metrics"]
        lines.append(
            f"{record['iteration']:>9} | {ops_summary:<58} | "
            f"{metrics['val_accuracy']:>12.4f} | {metrics['test_accuracy']:>13.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Executes the NAS-Bench-201 benchmark track."""
    load_dotenv()
    args = parse_args()
    agent_config = load_config(args.agent_config)
    nb_config = load_config(args.nasbench_config)
    override_configs = [load_config(path) for path in args.override]
    merged_config = merge_configs(agent_config, nb_config, *override_configs)

    seed = int(merged_config.get("training", {}).get("seed", 42))
    seed_everything(seed)

    nb_cfg = merged_config["nasbench201"]
    api_file = args.api_file or nb_cfg.get("api_file")

    experiment_cfg = merged_config["experiment"]
    output_dir = Path(experiment_cfg["output_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{experiment_cfg['name']}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(name="ra_nas_nb201", log_dir=str(experiment_dir))
    logger.info("Starting NAS-Bench-201 run at %s", experiment_dir)
    save_config(merged_config, str(experiment_dir / "config.yaml"))

    benchmark = NATSBenchLookup(
        api_file=api_file,
        dataset=str(nb_cfg.get("dataset", "cifar10-valid")),
        fast_mode=bool(nb_cfg.get("fast_mode", True)),
        verbose=bool(nb_cfg.get("verbose", False)),
    )

    generator = NB201ArchitectureGenerator(seed=seed)
    memory = ExperimentMemory()
    agent_cfg = merged_config.get("agent", {})
    prompt_builder = NB201PromptBuilder(
        top_k=int(agent_cfg.get("top_k_memory", 5)),
        self_correction=bool(agent_cfg.get("self_correction", True)),
    )
    agent = LLMAgent(
        agent_config=merged_config,
        search_space=generator,
        memory=memory,
        prompt_builder=prompt_builder,
    )

    num_iterations = int(args.iterations if args.iterations is not None else agent_cfg.get("max_iterations", 20))

    results = run(
        agent=agent,
        generator=generator,
        benchmark=benchmark,
        memory=memory,
        num_iterations=num_iterations,
        explore_every=int(agent_cfg.get("explore_every", 2)),
        screening_enabled=bool(nb_cfg.get("screening_enabled", True)),
        screen_hp=str(nb_cfg.get("screen_hp", "12")),
        full_hp=str(nb_cfg.get("full_hp", "200")),
        logger=logger,
    )

    memory.save(str(experiment_dir / "memory.json"))
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(build_summary_table(results))
    best = max(results, key=lambda r: r["metrics"]["val_accuracy"])
    logger.info(
        "Best architecture: ops=%s val_acc=%.4f test_acc=%.4f",
        best["arch"]["ops"], best["metrics"]["val_accuracy"], best["metrics"]["test_accuracy"],
    )
    logger.info("NAS-Bench-201 run complete. Results saved to %s", experiment_dir)


if __name__ == "__main__":
    main()
