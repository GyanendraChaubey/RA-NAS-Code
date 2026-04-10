"""Training loop implementation for RA-NAS architecture candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchvision.transforms import RandAugment
from tqdm import tqdm

from src.evaluation.metrics import accuracy
from src.training.early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def cutout(img: torch.Tensor, n_holes: int = 1, length: int = 16) -> torch.Tensor:
    """Applies CutOut regularisation: randomly masks square patches in a batch.

    Args:
        img: Float tensor of shape (N, C, H, W), values in [0, 1].
        n_holes: Number of patches to cut per image.
        length: Side length of each square patch.

    Returns:
        torch.Tensor: Augmented tensor with masked regions zeroed out.
    """
    h, w = img.size(2), img.size(3)
    mask = torch.ones_like(img)
    for _ in range(n_holes):
        cy = torch.randint(h, (1,)).item()
        cx = torch.randint(w, (1,)).item()
        y1 = max(0, cy - length // 2)
        y2 = min(h, cy + length // 2)
        x1 = max(0, cx - length // 2)
        x2 = min(w, cx + length // 2)
        mask[:, :, y1:y2, x1:x2] = 0.0
    return img * mask


def mixup_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Applies Mixup augmentation to a batch.

    Args:
        inputs: Input tensor (N, C, H, W).
        targets: Integer target tensor (N,).
        alpha: Beta distribution parameter.

    Returns:
        Tuple of (mixed_inputs, targets_a, targets_b, lam).
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[index]
    return mixed, targets, targets[index], lam


class Trainer:
    """Trains a candidate architecture with configurable optimization settings.

    Trainer is intentionally model-agnostic and fully config-driven so it can
    be reused across RA-NAS variants and ablation settings.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str,
        experiment_dir: Path,
        logger: Any,
    ) -> None:
        """Initializes trainer state and optimization components.

        Args:
            model: Model to train.
            config: Merged config dictionary.
            device: Target device string.
            experiment_dir: Directory for checkpoints.
            logger: Logger instance.
        """
        self.model = model
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        training_cfg = config["training"]
        self.epochs = int(training_cfg["epochs"])
        self.learning_rate = float(training_cfg["learning_rate"])
        self.weight_decay = float(training_cfg["weight_decay"])
        self.scheduler_type = str(training_cfg.get("scheduler", "none")).lower()
        self.save_best_only = bool(config["experiment"].get("save_best_only", True))
        self.warmup_epochs = int(training_cfg.get("warmup_epochs", 5))

        # Augmentation flags (on by default, disable in config for ablations)
        aug_cfg = training_cfg.get("augmentation", {})
        self.use_cutout = bool(aug_cfg.get("cutout", True))
        self.cutout_length = int(aug_cfg.get("cutout_length", 16))
        self.use_mixup = bool(aug_cfg.get("mixup", True))
        self.mixup_alpha = float(aug_cfg.get("mixup_alpha", 0.2))
        self.use_randaugment = bool(aug_cfg.get("randaugment", True))
        if self.use_randaugment:
            self._randaugment = RandAugment(num_ops=2, magnitude=9)
        else:
            self._randaugment = None

        # Label smoothing
        label_smoothing = float(training_cfg.get("label_smoothing", 0.1))

        # SWA
        swa_cfg = training_cfg.get("swa", {})
        self.use_swa = bool(swa_cfg.get("enabled", True))
        self.swa_start_frac = float(swa_cfg.get("start_frac", 0.75))  # start SWA at 75% of training
        self.swa_lr = float(swa_cfg.get("lr", 0.05))
        self._swa_model: AveragedModel | None = None
        self._swa_scheduler: SWALR | None = None

        optimizer_type = str(training_cfg.get("optimizer", "sgd")).lower()
        if optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=float(training_cfg.get("momentum", 0.9)),
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.scheduler = self._build_scheduler()
        self.monitor_mode = str(config.get("early_stopping", {}).get("mode", "max"))
        self.best_score = float("-inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch = 0

        es_cfg = config.get("early_stopping", {})
        self.early_stopping = None
        if bool(es_cfg.get("enabled", False)):
            self.early_stopping = EarlyStopping(
                patience=int(es_cfg["patience"]),
                mode=str(es_cfg["mode"]),
                monitor=str(es_cfg["monitor"]),
            )

    def _build_scheduler(self) -> Any:
        """Builds learning-rate scheduler from config.

        Supports cosine annealing with optional linear warmup, step decay, or none.

        Returns:
            Any: Scheduler object or None.

        Raises:
            ValueError: If scheduler type is unsupported.
        """
        if self.scheduler_type == "none":
            return None
        if self.scheduler_type in ("cosine", "cosine_warmup"):
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.epochs - self.warmup_epochs),
                eta_min=1e-5,
            )
            if self.warmup_epochs > 0:
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=self.warmup_epochs,
                )
                return SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[self.warmup_epochs],
                )
            return cosine
        if self.scheduler_type == "step":
            step_size = max(1, self.epochs // 3)
            return StepLR(self.optimizer, step_size=step_size, gamma=0.5)
        raise ValueError(
            f"Unsupported scheduler '{self.scheduler_type}'. "
            "Use one of: cosine, cosine_warmup, step, none."
        )

    def train(self, train_loader: Any, val_loader: Any) -> Dict[str, Any]:
        """Runs the complete training loop for one architecture.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.

        Returns:
            Dict[str, Any]: Final and historical training metrics.
        """
        # Initialise SWA components here (after optimizer is built)
        swa_start_epoch = max(1, int(self.epochs * self.swa_start_frac))
        if self.use_swa:
            self._swa_model = AveragedModel(self.model)
            self._swa_scheduler = SWALR(
                self.optimizer, swa_lr=self.swa_lr,
                anneal_epochs=max(1, self.epochs - swa_start_epoch),
            )

        history = []
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._val_epoch(val_loader)

            if self.use_swa and epoch >= swa_start_epoch:
                self._swa_model.update_parameters(self.model)
                self._swa_scheduler.step()
            elif self.scheduler is not None:
                self.scheduler.step()

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            history.append(epoch_metrics)
            self._save_checkpoint(epoch=epoch, metrics=epoch_metrics)

            self.logger.info(
                (
                    "Epoch %d/%d | train_loss=%.4f train_acc=%.4f "
                    "val_loss=%.4f val_acc=%.4f"
                ),
                epoch,
                self.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if self.early_stopping and self.early_stopping.step(epoch_metrics):
                self.logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        # Finalise SWA: update BatchNorm stats with averaged weights
        if self.use_swa and self._swa_model is not None:
            update_bn(train_loader, self._swa_model, device=self.device)
            # Swap model weights to SWA weights for downstream evaluation
            self.model.load_state_dict(self._swa_model.module.state_dict())
            self.logger.info("SWA weights applied.")

        last = history[-1]
        return {
            "train_loss": float(last["train_loss"]),
            "train_accuracy": float(last["train_accuracy"]),
            "val_loss": float(last["val_loss"]),
            "val_accuracy": float(last["val_accuracy"]),
            "best_epoch": int(self.best_epoch),
            "history": history,
        }

    def _train_epoch(self, loader: Any) -> Tuple[float, float]:
        """Runs one training epoch with optional CutOut and Mixup augmentation.

        Args:
            loader: Training dataloader.

        Returns:
            Tuple[float, float]: Average loss and top-1 accuracy.
        """
        self.model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0

        progress = tqdm(loader, desc="train", leave=False)
        for inputs, targets in progress:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # RandAugment (applied per-sample on the GPU tensor batch)
            if self._randaugment is not None:
                # RandAugment expects uint8 tensors; convert, augment, reconvert
                imgs_u8 = (inputs * 255).clamp(0, 255).byte()
                imgs_u8 = torch.stack([self._randaugment(img) for img in imgs_u8])
                inputs = imgs_u8.float() / 255.0

            # CutOut: zero out random square patches
            if self.use_cutout:
                inputs = cutout(inputs, n_holes=1, length=self.cutout_length)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_mixup:
                mixed, targets_a, targets_b, lam = mixup_batch(inputs, targets, self.mixup_alpha)
                outputs = self.model(mixed)
                loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = targets.size(0)
            acc = accuracy(outputs, targets, topk=(1,))["top1"]
            running_loss += float(loss.item()) * batch_size
            running_correct += (acc / 100.0) * batch_size
            running_total += batch_size

        avg_loss = running_loss / max(1, running_total)
        avg_acc = (running_correct / max(1, running_total)) * 100.0
        return avg_loss, avg_acc

    def _val_epoch(self, loader: Any) -> Tuple[float, float]:
        """Runs one validation epoch.

        Args:
            loader: Validation dataloader.

        Returns:
            Tuple[float, float]: Average loss and top-1 accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                batch_size = targets.size(0)
                acc = accuracy(outputs, targets, topk=(1,))["top1"]
                running_loss += float(loss.item()) * batch_size
                running_correct += (acc / 100.0) * batch_size
                running_total += batch_size

        avg_loss = running_loss / max(1, running_total)
        avg_acc = (running_correct / max(1, running_total)) * 100.0
        return avg_loss, avg_acc

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Saves checkpoint according to monitored metric and policy.

        Args:
            epoch: Current epoch index.
            metrics: Current epoch metrics dictionary.
        """
        monitor = str(self.config.get("early_stopping", {}).get("monitor", "val_accuracy"))
        current = float(metrics.get(monitor, float("-inf")))
        improved = (
            current > self.best_score
            if self.monitor_mode == "max"
            else current < self.best_score
        )

        if improved:
            self.best_score = current
            self.best_epoch = epoch

        if self.save_best_only and not improved:
            return

        checkpoint = {
            "epoch": epoch,
            "metrics": metrics,
            "arch_config": getattr(self.model, "arch_config", {}),
            "state_dict": self.model.state_dict(),
        }
        checkpoint_path = self.experiment_dir / "model.pt"
        torch.save(checkpoint, checkpoint_path)
