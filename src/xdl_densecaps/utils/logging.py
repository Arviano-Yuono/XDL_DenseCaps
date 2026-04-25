"""Logging helpers for XDL DenseCaps scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the project namespace."""

    return logging.getLogger(name)


def configure_logging(
    output_dir: str | Path | None = None,
    *,
    level: str | int = "INFO",
    log_file: str | Path | None = "train.log",
    use_console: bool = True,
) -> Path | None:
    """Configure project logging and optionally write logs to ``output_dir``.

    The configuration is scoped to the ``xdl_densecaps`` logger so external
    libraries keep their own logging behavior.
    """

    logger = logging.getLogger("xdl_densecaps")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(_coerce_log_level(level))
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logger.level)
        logger.addHandler(console_handler)

    log_path = _resolve_log_path(output_dir, log_file)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logger.level)
        logger.addHandler(file_handler)

    return log_path


def _coerce_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    normalized_level = level.upper()
    if normalized_level not in LOG_LEVELS:
        valid_levels = ", ".join(LOG_LEVELS)
        raise ValueError(f"Unknown log level: {level}. Choose one of: {valid_levels}.")
    return LOG_LEVELS[normalized_level]


def _resolve_log_path(output_dir: str | Path | None, log_file: str | Path | None) -> Path | None:
    if output_dir is None or log_file in (None, ""):
        return None

    log_file_path = Path(log_file)
    if log_file_path.is_absolute():
        return log_file_path
    return Path(output_dir) / log_file_path
