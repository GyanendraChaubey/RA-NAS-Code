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

        # Multi-fidelity: screen with fewer epochs, then full-train top candidates
        training_cfg = config.get("training", {})
        self.screening_epochs: int = int(training_cfg.get("screening_epochs", 0))
        self.full_epochs: int = int(training_cfg.get("epochs", 50))
        self._screening_buffer: List[Dict[str, Any]] = []

    def run(self, num_iterations: int) -> List[Dict[str, Any]]:
        """Executes the iterative RA-NAS optimization loop with optional multi-fidelity screening.

        When ``screening_epochs > 0``, every two consecutive iterations are paired:
        both architectures are screened quickly, then only the better one is
        fully trained. This allows twice as many architectures to be explored
        in the same wall-clock time.
        """
        memory_save_path = self.experiment_dir / "memory.json"
        self.results = []
        candidate_arch = None

        for iteration in range(1, num_iterations + 1):
            if (
                candidate_arch is None
                or (self.explore_every > 0 and iteration % self.explore_every == 0)
            ):
                arch = self.agent.propose_architecture()
            else:
                arch = candidate_arch
            self.generator.validate(arch)

            iteration_dir = self.experiment_dir / f"iter_{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            # ── Multi-fidelity screening ──────────────────────────────────────
            if self.screening_epochs > 0:
                arch, train_metrics, eval_metrics = self._screened_train(
                    arch=arch,
                    iteration=iteration,
                    iteration_dir=iteration_dir,
                )
            else:
                model = build_model(arch_config=arch, num_classes=self.num_classes, device=self.device)
                trainer = self.trainer_factory(model, iteration_dir)
                train_metrics = trainer.train(train_loader=self.train_loader, val_loader=self.val_loader)
                evaluator = self.evaluator_factory(model)
                eval_metrics = evaluator.evaluate(self.val_loader)
            # ─────────────────────────────────────────────────────────────────

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

            self.memory.add(arch, combined_metrics, predicted_accuracy=getattr(self.agent, "_last_predicted_accuracy", None))
            self.memory.save(str(memory_save_path))

            candidate_arch = self.agent.refine_architecture(arch=arch, feedback=combined_metrics)
            self.generator.validate(candidate_arch)

            record = {"iteration": iteration, "arch": arch, "metrics": combined_metrics}
            self.results.append(record)
            self._log_iteration(iteration=iteration, arch=arch, metrics=combined_metrics)
            self.logger.info("Completed NAS iteration %d/%d", iteration, num_iterations)

        return self.results

    def _screened_train(
        self,
        arch: Dict[str, Any],
        iteration: int,
        iteration_dir: Path,
    ):
        """Screens current arch against a buffered previous one; full-trains the winner.

        Phase A: screen current arch with self.screening_epochs.
        Phase B: on even iterations, compare with buffered arch and full-train the winner.

        Returns:
            Tuple of (winning_arch, train_metrics, eval_metrics).
        """
        screen_config = {
            **self.config,
            "training": {**self.config.get("training", {}), "epochs": self.screening_epochs},
        }
        model = build_model(arch_config=arch, num_classes=self.num_classes, device=self.device)
        screen_trainer = self.trainer_factory(model, iteration_dir, screen_config)
        screen_metrics = screen_trainer.train(train_loader=self.train_loader, val_loader=self.val_loader)
        screen_val_acc = screen_metrics["val_accuracy"]

        self._screening_buffer.append({
            "arch": arch,
            "screen_val_acc": screen_val_acc,
            "iteration_dir": iteration_dir,
        })

        # Every 2 iterations: pick winner and full-train
        if len(self._screening_buffer) >= 2:
            a, b = self._screening_buffer[-2], self._screening_buffer[-1]
            winner = a if a["screen_val_acc"] >= b["screen_val_acc"] else b
            self.logger.info(
                "Screening: iter %d acc=%.4f vs iter %d acc=%.4f — full-training %s option (acc=%.4f)",
                iteration - 1, a["screen_val_acc"],
                iteration, b["screen_val_acc"],
                "previous" if winner is a else "current",
                winner["screen_val_acc"],
            )
            self._screening_buffer.clear()
            win_arch = winner["arch"]
            win_dir = winner["iteration_dir"]
            win_model = build_model(arch_config=win_arch, num_classes=self.num_classes, device=self.device)
            full_trainer = self.trainer_factory(win_model, win_dir)
            train_metrics = full_trainer.train(train_loader=self.train_loader, val_loader=self.val_loader)
            evaluator = self.evaluator_factory(win_model)
            eval_metrics = evaluator.evaluate(self.val_loader)
            return win_arch, train_metrics, eval_metrics
        else:
            # Odd iteration: screened only — return screening metrics directly
            # (no extra full-train; saves wall-clock so more archs can be explored)
            evaluator = self.evaluator_factory(model)
            eval_metrics = evaluator.evaluate(self.val_loader)
            return arch, screen_metrics, eval_metrics

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
