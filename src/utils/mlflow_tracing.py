#!/usr/bin/env python
"""MLflow tracing helpers for session-scoped trace grouping and per-document
nested span trees spanning multiple LangGraph tasks."""

from contextlib import contextmanager
from typing import Any, Generator


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


@contextmanager
def _no_ambient_span() -> Generator[None, None, None]:
    """Suppress whatever span MLflow considers "currently active" for the
    duration of this block, so a new root span created inside it is a true,
    independent root rather than implicitly nesting under it.

    Needed because ``mlflow.start_span_no_context(name)`` with no explicit
    ``parent_span`` does *not* ignore ambient context -- verified empirically
    that it silently attaches to whatever span is active (e.g. one created by
    ``mlflow.langchain.autolog()``'s own instrumentation of the LangGraph task
    that's calling this). When multiple documents' tasks share that ambient
    span, their supposedly-independent root spans all collapse into one trace.

    This clears MLflow's own internal tracing context (separate from, and not
    cleared by, OpenTelemetry's standard ``context_api.attach``/``detach`` --
    verified that clearing the OTel context alone does *not* fix this). It's
    an internal API (``mlflow.tracing.provider.mlflow_runtime_context``), not
    part of MLflow's public surface, so this degrades gracefully (falls back
    to the ambient-nesting behavior rather than erroring) if a future MLflow
    version removes or renames it.
    """
    try:
        from mlflow.tracing.provider import mlflow_runtime_context
        from opentelemetry.context import Context
    except ImportError:
        yield
        return

    token = mlflow_runtime_context.attach(Context())
    try:
        yield
    finally:
        mlflow_runtime_context.detach(token)


def start_document_span(run_id: str | None, name: str, **attributes: Any):
    """Start a new root span (not context-managed) for one document's journey
    through the pipeline, attributed to ``run_id``.

    Returns ``None`` if ``run_id`` is missing (tracing disabled). Otherwise
    returns a ``mlflow.entities.LiveSpan`` that the caller must eventually pass
    to ``end_span`` exactly once -- on every exit path for that document, not
    just the success path -- since nothing else will close it.

    Unlike ``mlflow.start_span()``, this isn't a context manager: the whole
    point is that this span's lifetime spans multiple independently-scheduled
    LangGraph tasks (convert -> chunk -> embed -> store), so it can't be tied
    to a single ``with`` block the way a normal span is.
    """
    if not run_id:
        return None
    import mlflow

    with run_context(run_id), _no_ambient_span():
        return mlflow.start_span_no_context(
            name, attributes=attributes or None
        )


def start_child_span(
    run_id: str | None, parent_span: Any, name: str, **attributes: Any
):
    """Start a child span of ``parent_span`` (not context-managed), attributed
    to ``run_id``.

    Returns ``None`` if ``run_id`` is missing or ``parent_span`` is ``None``
    (e.g. tracing disabled, or the document's root span already ended because
    an earlier stage failed). Caller must ``end_span`` the result.
    """
    if not run_id or parent_span is None:
        return None
    import mlflow

    with run_context(run_id):
        return mlflow.start_span_no_context(
            name, parent_span=parent_span, attributes=attributes or None
        )


def end_span(span: Any, *, error: str | None = None, **attributes: Any) -> None:
    """End a span started by ``start_document_span``/``start_child_span``.

    No-op if ``span`` is ``None``. Sets ``attributes`` on the span, marks it
    ``ERROR`` (recording ``error`` as an attribute) or ``OK``, then ends it.
    """
    if span is None:
        return
    for key, value in attributes.items():
        span.set_attribute(key, value)
    if error:
        span.set_attribute("error", error)
    span.end(status="ERROR" if error else "OK")


__all__ = [
    "session_context",
    "run_context",
    "start_document_span",
    "start_child_span",
    "end_span",
]
