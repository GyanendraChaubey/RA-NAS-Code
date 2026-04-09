"""Config loading, merging, and persistence helpers for RA-NAS."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        path: YAML file path.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If YAML content is not a dictionary.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {cfg_path} must be a mapping.")
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merges multiple config dictionaries.

    Later config entries override earlier values.

    Args:
        *configs: Any number of config dictionaries.

    Returns:
        Dict[str, Any]: Deep-merged result.
    """

    def merge(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        result = deepcopy(left)
        for key, value in right.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    merged: Dict[str, Any] = {}
    for cfg in configs:
        if cfg:
            merged = merge(merged, cfg)
    return merged


def save_config(config: Dict[str, Any], path: str) -> None:
    """Saves configuration dictionary to YAML.

    Args:
        config: Configuration dictionary.
        path: Output YAML file path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

