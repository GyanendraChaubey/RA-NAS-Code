"""Search-space definition and architecture validation for RA-NAS."""

from __future__ import annotations

from typing import Any, Dict, List

SEARCH_SPACE: Dict[str, Any] = {
    "num_layers": {"min": 2, "max": 8},
    "filters_per_layer": [16, 32, 64, 128, 256],
    "kernel_sizes": [3, 5],
    "activations": ["relu", "gelu", "silu"],
    "use_batchnorm": [True, False],
    "use_dropout": [True, False],
    "dropout_rate": {"min": 0.0, "max": 0.6},
    "use_skip_connections": [True, False],
    "pooling": ["max", "avg"],
}


def _assert(condition: bool, message: str) -> None:
    """Raises ValueError if condition is false.

    Args:
        condition: Condition to verify.
        message: Exception message when condition fails.
    """
    if not condition:
        raise ValueError(message)


def _valid_filter_values(constraints: Dict[str, Any]) -> List[int]:
    """Computes valid filter values from global and config constraints.

    Args:
        constraints: Config-level architecture constraints.

    Returns:
        List[int]: Sorted list of valid filter sizes.
    """
    min_filters = int(constraints.get("min_filters", min(SEARCH_SPACE["filters_per_layer"])))
    max_filters = int(constraints.get("max_filters", max(SEARCH_SPACE["filters_per_layer"])))
    return [
        value
        for value in SEARCH_SPACE["filters_per_layer"]
        if min_filters <= value <= max_filters
    ]


def validate_architecture(arch: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    """Validates an architecture dict against schema and constraints.

    Args:
        arch: Candidate architecture dictionary.
        constraints: Config-level constraints from train config.

    Returns:
        bool: True if architecture is valid.

    Raises:
        ValueError: If architecture does not satisfy schema or constraints.
    """
    required_keys = {
        "num_layers",
        "filters",
        "kernels",
        "activation",
        "use_batchnorm",
        "use_dropout",
        "dropout_rate",
        "use_skip_connections",
        "pooling",
    }
    missing = required_keys.difference(arch.keys())
    _assert(not missing, f"Architecture missing required keys: {sorted(missing)}")

    num_layers = arch["num_layers"]
    _assert(isinstance(num_layers, int), "num_layers must be an int.")
    min_layers = int(
        constraints.get("min_layers", SEARCH_SPACE["num_layers"]["min"])
    )
    max_layers = int(
        constraints.get("max_layers", SEARCH_SPACE["num_layers"]["max"])
    )
    _assert(min_layers <= num_layers <= max_layers, f"num_layers must be in [{min_layers}, {max_layers}].")

    filters = arch["filters"]
    kernels = arch["kernels"]
    _assert(isinstance(filters, list), "filters must be a list.")
    _assert(isinstance(kernels, list), "kernels must be a list.")
    _assert(len(filters) == num_layers, "filters length must equal num_layers.")
    _assert(len(kernels) == num_layers, "kernels length must equal num_layers.")

    valid_filters = _valid_filter_values(constraints)
    _assert(valid_filters, "No valid filter values available after applying constraints.")
    for value in filters:
        _assert(isinstance(value, int), f"Filter value must be int, got {value!r}.")
        _assert(
            value in valid_filters,
            f"Filter value {value} is invalid. Allowed values: {valid_filters}.",
        )

    allowed_kernels = constraints.get("allowed_kernels", SEARCH_SPACE["kernel_sizes"])
    for kernel in kernels:
        _assert(kernel in SEARCH_SPACE["kernel_sizes"], f"Kernel {kernel} not in global search space.")
        _assert(kernel in allowed_kernels, f"Kernel {kernel} not allowed by constraints.")

    activation = arch["activation"]
    allowed_activations = constraints.get("allowed_activations", SEARCH_SPACE["activations"])
    _assert(activation in SEARCH_SPACE["activations"], f"Activation {activation} not in global search space.")
    _assert(activation in allowed_activations, f"Activation {activation} not allowed by constraints.")

    _assert(isinstance(arch["use_batchnorm"], bool), "use_batchnorm must be a bool.")
    _assert(isinstance(arch["use_dropout"], bool), "use_dropout must be a bool.")
    _assert(isinstance(arch["use_skip_connections"], bool), "use_skip_connections must be a bool.")
    _assert(arch["pooling"] in SEARCH_SPACE["pooling"], f"pooling must be one of {SEARCH_SPACE['pooling']}.")

    dropout_rate = float(arch["dropout_rate"])
    _assert(
        SEARCH_SPACE["dropout_rate"]["min"] <= dropout_rate <= SEARCH_SPACE["dropout_rate"]["max"],
        (
            "dropout_rate must be in "
            f"[{SEARCH_SPACE['dropout_rate']['min']}, {SEARCH_SPACE['dropout_rate']['max']}]."
        ),
    )
    if not arch["use_dropout"]:
        _assert(dropout_rate == 0.0, "dropout_rate must be 0.0 when use_dropout=False.")

    return True

