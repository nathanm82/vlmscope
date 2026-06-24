"""Logging helpers.

A single process-wide logger named ``vlmscope``. The level is read once from
the ``VLMSCOPE_LOG_LEVEL`` environment variable so library users can stay quiet
by default and opt into detail.
"""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def get_logger(name: str = "vlmscope") -> logging.Logger:
    """Return a configured logger, attaching a stream handler exactly once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        level = os.environ.get("VLMSCOPE_LOG_LEVEL", "WARNING").upper()
        logger.setLevel(level)
        logger.propagate = False
    return logger
