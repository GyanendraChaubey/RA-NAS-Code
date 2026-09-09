#!/usr/bin/env python3
"""Aggregates results across multiple RA-NAS experiment runs into one comparison table.

Intended for ablation studies: run each variant (see configs/ablations/ and the
README's Ablation Study Guide), then run this script to fold every completed
experiments/<name>_<timestamp>/ run into a single CSV — accuracy, cost, and which
ablation flags were on — ready to paste into a paper's results table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI args.
    """
    parser = argparse.ArgumentParser(description="Aggregate RA-NAS experiment results into a comparison table.")
    parser.add_argument("--experiments-dir", type=str, default="experiments")
    parser.add_argument("--out-csv", type=str, default="experiments/ablation_summary.csv")
    return parser.parse_args()


def _flag_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts the ablation-relevant flags from a run's saved config.yaml.

    Args:
        config: Parsed config.yaml contents for one run.

    Returns:
        Dict[str, Any]: Which track ran and which ablation toggles were set.
    """
    agent_cfg = config.get("agent", {})
    training_cfg = config.get("training", {})
    nb_cfg = config.get("nasbench201")
    return {
        "track": "nasbench201" if nb_cfg is not None else "main",
        "mock_mode": agent_cfg.get("mock_mode"),
        "explore_every": agent_cfg.get("explore_every"),
        "diversity_penalty": agent_cfg.get("diversity_penalty"),
        "self_correction": agent_cfg.get("self_correction"),
        "screening_enabled": (
            nb_cfg.get("screening_enabled")
            if nb_cfg is not None
            else bool(training_cfg.get("screening_epochs", 0))
        ),
    }


def summarize_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Builds one summary row for a single experiment directory.

    Args:
        run_dir: Path to one experiments/<name>_<timestamp>/ directory.

    Returns:
        Optional[Dict[str, Any]]: A summary row, or None if the directory doesn't
            contain a completed run (missing/empty metrics.json or config.yaml).
    """
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists() or not config_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as file:
        results: List[Dict[str, Any]] = json.load(file)
    if not results:
        return None
    config = load_config(str(config_path))

    best = max(results, key=lambda r: r["metrics"]["val_accuracy"])
    last_cost = results[-1].get("cost", {})

    row: Dict[str, Any] = {
        "run": run_dir.name,
        "iterations": len(results),
        "best_iteration": best["iteration"],
        "best_val_accuracy": round(float(best["metrics"]["val_accuracy"]), 4),
        "best_test_accuracy": (
            round(float(best["metrics"]["test_accuracy"]), 4) if "test_accuracy" in best["metrics"] else ""
        ),
        "total_llm_calls": last_cost.get("total_llm_calls", ""),
        "estimated_cost_usd": last_cost.get("estimated_cost_usd", ""),
    }
    row.update(_flag_summary(config))
    return row


def main() -> None:
    """Scans all experiment directories and writes a combined summary CSV."""
    args = parse_args()
    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"No experiments directory found at {experiments_dir}.")
        return

    rows = []
    for run_dir in sorted(experiments_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"No completed runs found under {experiments_dir}.")
        return

    fieldnames = list(rows[0].keys())
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = " | ".join(fieldnames)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row[key]) for key in fieldnames))
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
