"""NATS-Bench (NAS-Bench-201 topology space) lookup-based architecture evaluator.

Replaces real training with an instant table lookup against precomputed
results, so RA-NAS's full propose-refine-memory loop can be benchmarked
against a standardized, reproducible ground truth instead of a custom-trained
model — and directly compared against numbers reported by other LLM-NAS
papers (e.g. LLMatic, RZ-NAS) that also evaluate on this benchmark.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.nasbench201.search_space import ops_to_arch_str


class NATSBenchLookup:
    """Wraps the NATS-Bench API for instant, precomputed architecture evaluation."""

    def __init__(
        self,
        api_file: Optional[str],
        dataset: str = "cifar10-valid",
        fast_mode: bool = True,
        verbose: bool = False,
    ) -> None:
        """Loads the NATS-Bench topology-search-space (tss) API.

        Args:
            api_file: Path to the downloaded NATS-tss benchmark file/archive,
                or None to look it up under $TORCH_HOME (see NATS-Bench docs).
            dataset: One of "cifar10-valid", "cifar10", "cifar100", "ImageNet16-120".
                "cifar10-valid" is the standard NAS-search protocol split
                (search on a held-out validation split, report on the true test set).
            fast_mode: If True, loads the smaller per-architecture archive on demand
                instead of the full pickle (recommended; much lower memory use).
            verbose: If True, the underlying API logs every query.

        Raises:
            ImportError: If the `nats_bench` package is not installed.
        """
        try:
            from nats_bench import create
        except ImportError as error:
            raise ImportError(
                "nats_bench is required for the NAS-Bench-201 track. Install it with "
                "`pip install nats_bench` and download the topology (tss) benchmark file "
                "per https://github.com/D-X-Y/NATS-Bench#preparation-and-download."
            ) from error
        self.dataset = dataset
        self._api = create(api_file, "tss", fast_mode=fast_mode, verbose=verbose)

    def evaluate(self, arch: Dict[str, Any], hp: str) -> Dict[str, Any]:
        """Looks up precomputed metrics for an architecture at a given epoch budget.

        Args:
            arch: Architecture dict with an "ops" key (6 op names).
            hp: Epoch-budget key recognised by NATS-Bench ("12" for the cheap
                multi-fidelity proxy, "200" for the full-fidelity result).

        Returns:
            Dict[str, Any]: train/val/test accuracy and loss, param/FLOP/latency
                cost, and the epoch budget queried (best_epoch).
        """
        arch_str = ops_to_arch_str(arch["ops"])
        info = self._api.get_more_info(arch_str, self.dataset, hp=hp, is_random=True)
        cost = self._api.get_cost_info(arch_str, self.dataset, hp=hp)
        return {
            "train_loss": float(info.get("train-loss", 0.0)),
            "train_accuracy": float(info.get("train-accuracy", 0.0)),
            "val_loss": float(info.get("valid-loss", 0.0)),
            "val_accuracy": float(info.get("valid-accuracy", 0.0)),
            "test_accuracy": float(info.get("test-accuracy", 0.0)),
            "num_params": float(cost.get("params", 0.0)),
            "flops": float(cost.get("flops", 0.0)),
            "inference_time_ms": float(cost.get("latency", 0.0)) * 1000.0,
            "best_epoch": int(hp),
        }
