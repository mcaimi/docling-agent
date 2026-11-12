#!/usr/bin/env python
"""Logging configuration: plain text for terminals, JSON lines for containers."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Attributes every LogRecord has; anything else on a record came from ``extra=``.
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None))
) | {"message", "asctime", "taskName"}

# Package loggers (``__main__`` when run as ``python -m src.docling_agent``).
_APP_LOGGERS = ("src", "__main__")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with ``extra=`` fields and static context merged in."""

    def __init__(self, static_fields: dict[str, Any] | None = None):
        super().__init__()
        self._static = {k: v for k, v in (static_fields or {}).items() if v is not None}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **self._static,
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    fmt: str = "text",
    static_fields: dict[str, Any] | None = None,
) -> None:
    """Configure the root logger for CLI/container use.

    ``fmt="text"`` keeps the historical console output (bare messages on
    stdout); ``fmt="json"`` emits JSON lines with ``static_fields`` (e.g.
    shard index, scheduler run id) attached to every record.
    """
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter(static_fields))
    else:
        handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    # Third-party libraries (docling, httpx, mlflow) are chatty at INFO; only
    # this package's loggers follow ``level``.
    root.setLevel(logging.WARNING)
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(getattr(logging, str(level).upper(), logging.INFO))


__all__ = ["configure_logging", "JsonFormatter"]
