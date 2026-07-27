"""Central logging configuration for the ``memories`` package."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    """Configure the ``memories`` package logger.

    Level resolution: the explicit ``level`` argument if given, else the ``LOG_LEVEL``
    environment variable, else ``INFO``. An unrecognised level name falls back to ``INFO``.
    Idempotent: a second call updates the level but does not attach a duplicate handler.
    """
    raw = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    resolved = raw.upper()
    numeric = logging.getLevelNamesMapping().get(resolved, logging.INFO)

    logger = logging.getLogger("memories")
    logger.setLevel(numeric)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
