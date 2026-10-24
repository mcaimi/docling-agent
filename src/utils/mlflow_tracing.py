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


@contextmanager
def run_context(run_id: str | None) -> Generator[None, None, None]:
    """Make ``run_id`` the active MLflow run for spans created in this scope.

    No-op when ``run_id`` is missing. Needed because LangGraph runs ``@task``
    functions in worker threads that don't inherit the active run from the
    thread that called ``mlflow.start_run()`` -- without this, every
    ``mlflow.start_span()`` call in a task finds no active run and MLflow
    creates a new orphan run to hold it. Passing ``run_id=`` directly to
    ``start_span()`` is not sufficient on its own: verified empirically that
    MLflow still spawns a new run in that case, even for a span that reports
    itself as root. Re-entering the run via ``mlflow.start_run(run_id=...)``
    first is what actually prevents it.

    Safe to nest: callers at different layers (e.g. a task function and a
    helper it calls) may both wrap their work in ``run_context(run_id)`` for
    the same id. Re-entering ``mlflow.start_run(run_id=...)`` for a run that's
    already active in the current context raises ("Run ... is already
    active"), so this checks first and no-ops if so.
    """
    if not run_id:
        yield
        return
    import mlflow

    active = mlflow.active_run()
    if active is not None and active.info.run_id == run_id:
        yield
        return

    with mlflow.start_run(run_id=run_id):
        yield


__all__ = ["session_context", "run_context"]
