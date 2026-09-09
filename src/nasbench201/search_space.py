"""Search-space definition for the NAS-Bench-201 topology cell.

The cell has 4 nodes (0=input, 3=output) and 6 directed edges:
edge1=(0->1), edge2=(0->2), edge3=(1->2), edge4=(0->3), edge5=(1->3), edge6=(2->3).
Each edge selects one of 5 fixed operations, giving 5**6 = 15,625 candidate cells —
this is the same space NATS-Bench and the original NAS-Bench-201 benchmark files index.
"""

from __future__ import annotations

from typing import Any, Dict, List

OPS: List[str] = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
NUM_EDGES = 6


def ops_to_arch_str(ops: List[str]) -> str:
    """Converts a 6-op list into NATS-Bench's architecture string format.

    Args:
        ops: List of 6 operation names, one per cell edge (edge1..edge6).

    Returns:
        str: Architecture string, e.g.
            "|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|".

    Raises:
        ValueError: If ops does not have exactly NUM_EDGES entries.
    """
    if len(ops) != NUM_EDGES:
        raise ValueError(f"Expected {NUM_EDGES} ops, got {len(ops)}.")
    return (
        f"|{ops[0]}~0|+"
        f"|{ops[1]}~0|{ops[2]}~1|+"
        f"|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
    )


def validate_ops(arch: Dict[str, Any]) -> bool:
    """Validates a NAS-Bench-201 architecture dict.

    Args:
        arch: Architecture dict expected to contain an "ops" key.

    Returns:
        bool: True if valid.

    Raises:
        ValueError: If "ops" is missing, wrong length, or contains an unknown op.
    """
    if "ops" not in arch:
        raise ValueError("Architecture missing required key: 'ops'.")
    ops = arch["ops"]
    if not isinstance(ops, list) or len(ops) != NUM_EDGES:
        raise ValueError(f"'ops' must be a list of {NUM_EDGES} operation names.")
    for op in ops:
        if op not in OPS:
            raise ValueError(f"Unknown operation {op!r}. Allowed: {OPS}.")
    return True
