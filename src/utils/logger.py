"""Structured logging utility for RA-NAS experiments."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_dir: str) -> logging.Logger:
    """Creates an experiment-scoped logger with console and file handlers.

    Args:
        name: Logger name.
        log_dir: Directory to store experiment.log.

    Returns:
        logging.Logger: Configured logger instance.
    """
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    file_handler = logging.FileHandler(directory / "experiment.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

