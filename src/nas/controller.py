"""NAS controller that orchestrates the RA-NAS propose-train-evaluate loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

from src.agents.memory import ExperimentMemory
from src.models.model_builder import build_model


class NASController:
    """Coordinates agent, generator, trainer, evaluator, and memory.

    This module is the core experiment loop that links architecture reasoning
    with model training/evaluation feedback over multiple iterations.
    """

    def __init__(
        self,
        agent: Any,
        generator: Any,
        trainer: Callable[..., Any],
        evaluator: Callable[..., Any],
        memory: ExperimentMemory,
        config: Dict[str, Any],
        train_loader: Any,
        val_loader: Any,
        device: str,
        logger: Any,
        experiment_dir: Any,
        num_classes: int,
        explore_every: int = 3,
    ) -> None:
        """Initializes NAS controller dependencies.

        Args:
            agent: Agent object with propose_architecture/refine_architecture.
            generator: Architecture generator with validation methods.
            trainer: Factory/callable creating a trainer from model and iter dir.
            evaluator: Factory/callable creating evaluator from model.
            memory: Shared experiment memory.
            config: Serialisable configuration dictionary (no runtime objects).
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            device: PyTorch device string.
            logger: Experiment logger instance.
            experiment_dir: Root directory for checkpoints and memory snapshots.
            num_classes: Number of target classes for model construction.
            explore_every: Force a fresh random proposal every this many iterations
                to avoid pure greedy local optima. Set to 0 to disable.
        """
        self.agent = agent
        self.generator = generator
        self.trainer_factory = trainer
        self.evaluator_factory = evaluator
        self.memory = memory
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.experiment_dir = Path(experiment_dir)
        self.num_classes = int(num_classes)
        self.explore_every = max(1, int(explore_every)) if explore_every > 0 else 0
        self.results: List[Dict[str, Any]] = []

    def run(self, num_iterations: int) -> List[Dict[str, Any]]:
        """Executes the iterative RA-NAS optimization loop.

        Args:
            num_iterations: Number of propose-train-evaluate rounds.

        Returns:
            List[Dict[str, Any]]: Per-iteration records.
        """
        memory_save_path = self.experiment_dir / "memory.json"
        self.results = []
        candidate_arch = None

        for iteration in range(1, num_iterations + 1):
            # Explore on iteration 1 and every explore_every iterations to avoid
            # converging to a local optimum through pure greedy refinement.
            if (
                candidate_arch is None
                or (self.explore_every > 0 and iteration % self.explore_every == 0)
            ):
                arch = self.agent.propose_architecture()
            else:
                arch = candidate_arch
            self.generator.validate(arch)

            model = build_model(
                arch_config=arch,
                num_classes=self.num_classes,
                device=self.device,
            )
            iteration_dir = self.experiment_dir / f"iter_{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            trainer = self.trainer_factory(model, iteration_dir)
            train_metrics = trainer.train(train_loader=self.train_loader, val_loader=self.val_loader)
            evaluator = self.evaluator_factory(model)
            eval_metrics = evaluator.evaluate(self.val_loader)

            combined_metrics = {
                "train_loss": train_metrics["train_loss"],
                "train_accuracy": train_metrics["train_accuracy"],
                "val_loss": eval_metrics["loss"],
                "val_accuracy": eval_metrics["accuracy"],
                "top5_accuracy": eval_metrics["top5_accuracy"],
                "num_params": eval_metrics["num_params"],
                "flops": eval_metrics["flops"],
                "inference_time_ms": eval_metrics["inference_time_ms"],
                "best_epoch": train_metrics["best_epoch"],
            }

            self.memory.add(arch, combined_metrics)
            self.memory.save(str(memory_save_path))

            candidate_arch = self.agent.refine_architecture(
                arch=arch,
                feedback=combined_metrics,
            )
            self.generator.validate(candidate_arch)

            record = {
                "iteration": iteration,
                "arch": arch,
                "metrics": combined_metrics,
            }
            self.results.append(record)
            self._log_iteration(iteration=iteration, arch=arch, metrics=combined_metrics)
            self.logger.info("Completed NAS iteration %d/%d", iteration, num_iterations)

        return self.results

    def _log_iteration(self, iteration: int, arch: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Logs a concise summary for a single NAS iteration.

        Args:
            iteration: Iteration index.
            arch: Architecture dictionary.
            metrics: Metrics dictionary for this iteration.
        """
        self.logger.info(
            "Iteration=%d | layers=%d act=%s pool=%s val_acc=%.4f val_loss=%.4f params=%d",
            iteration,
            arch["num_layers"],
            arch["activation"],
            arch["pooling"],
            float(metrics["val_accuracy"]),
            float(metrics["val_loss"]),
            int(metrics["num_params"]),
        )
