#!/usr/bin/env python
"""MLflow tracing helpers for session-scoped trace grouping."""

from contextlib import contextmanager
from typing import Generator


@contextmanager
def session_context(session_id: str | None) -> Generator[None, None, None]:
    """Apply MLflow session metadata to traces created in this scope.

    No-op when ``session_id`` is missing. Otherwise wraps
    ``mlflow.tracing.context(session_id=...)``.
    """
    if not session_id:
        yield
        return
    import mlflow

    with mlflow.tracing.context(session_id=session_id):
        yield


__all__ = ["session_context"]
