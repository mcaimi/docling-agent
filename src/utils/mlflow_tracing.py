#!/usr/bin/env python
"""MLflow tracing helpers for session-scoped trace grouping and per-document
nested span trees spanning multiple LangGraph tasks."""

import json
import sys
import threading
from contextlib import contextmanager
from typing import Any, Generator

# --- span payload guard -------------------------------------------------------
# Span inputs/outputs/attributes are uploaded to the MLflow tracking server with
# every trace. Embedding vectors, whole DoclingDocuments or full markdown can
# turn one trace into megabytes of JSON, which bogs down the server and breaks
# the agent's API calls to it. Traces must carry parameters and metrics only:
# the guard below reduces anything bulky to a short, human-readable summary.

# Keys whose values are bulky by nature: always summarized, never sent as-is.
_BULKY_KEYS = frozenset(
    {"docling_document", "embeddings", "embedding", "vectors", "content", "documents", "chunks"}
)
# Longest string sent verbatim; longer strings become a preview.
_MAX_STR_CHARS = 200
# Longest list sent element by element; longer lists are cut with a marker.
_MAX_LIST_ITEMS = 20
# A list of at least this many numbers is treated as a vector.
_MIN_VECTOR_LEN = 8
# Nesting depth after which values are replaced by their type name.
_MAX_DEPTH = 6
# Hard cap on a whole serialized payload (inputs, outputs or one attribute).
_MAX_PAYLOAD_CHARS = 8192


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_vector(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= _MIN_VECTOR_LEN
        and all(_is_number(v) for v in value)
    )


def _preview(text: str) -> str:
    if len(text) <= _MAX_STR_CHARS:
        return text
    return f"{text[:_MAX_STR_CHARS]}... <truncated, {len(text)} chars>"


def _summarize_bulky(value: Any) -> Any:
    """Short description of a value stored under a ``_BULKY_KEYS`` key."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _preview(value)
    if _is_vector(value):
        return f"<vector dim={len(value)}>"
    if isinstance(value, (list, tuple)):
        dims = {len(v) for v in value if _is_vector(v)}
        if dims:
            dim = dims.pop() if len(dims) == 1 else "mixed"
            return f"<{len(value)} vectors dim={dim}>"
        return f"<{len(value)} items>"
    if isinstance(value, dict):
        return f"<dict with {len(value)} keys>"
    return f"<{type(value).__name__}>"


def summarize_payload(value: Any, _depth: int = 0) -> Any:
    """Return a JSON-friendly copy of ``value`` without bulky data.

    Vectors become ``"<vector dim=N>"``, lists of vectors
    ``"<N vectors dim=D>"``, values under ``_BULKY_KEYS`` (documents,
    embeddings, markdown content, ...) a short summary, long strings a
    preview, long lists are cut, and arbitrary objects (e.g. a
    DoclingDocument) are replaced by their type name. Never mutates
    ``value``.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _preview(value)
    if _depth >= _MAX_DEPTH:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {
            str(k): (
                _summarize_bulky(v) if k in _BULKY_KEYS else summarize_payload(v, _depth + 1)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        if _is_vector(value):
            return f"<vector dim={len(value)}>"
        if value and all(_is_vector(v) for v in value):
            return _summarize_bulky(value)
        items = [summarize_payload(v, _depth + 1) for v in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            items.append(f"<{len(value) - _MAX_LIST_ITEMS} more items>")
        return items
    return f"<{type(value).__name__}>"


def _cap(value: Any) -> Any:
    """Replace ``value`` by a preview if its JSON form exceeds the payload cap."""
    try:
        encoded = json.dumps(value, default=str)
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"
    if len(encoded) <= _MAX_PAYLOAD_CHARS:
        return value
    return f"{encoded[:_MAX_PAYLOAD_CHARS]}... <truncated, {len(encoded)} chars>"


def payload_guard(span: Any) -> None:
    """MLflow span processor: shrink inputs, outputs and attributes before export.

    Registered with ``mlflow.tracing.configure(span_processors=[payload_guard])``
    when tracing starts (see ``src/docling_agent.py``). It's a safety net: the
    pipeline only records counts and parameters on its own spans, but this
    guarantees nothing bulky reaches the tracking server even if a future span
    records a vector, a document or a whole chunk list.
    """
    inputs = span.inputs
    if inputs is not None:
        span.set_inputs(_cap(summarize_payload(inputs)))
    outputs = span.outputs
    if outputs is not None:
        span.set_outputs(_cap(summarize_payload(outputs)))
    for key, value in list((span.attributes or {}).items()):
        # MLflow keeps its own bookkeeping (trace id, span type, inputs and
        # outputs) in "mlflow."-prefixed attributes; inputs/outputs were
        # already handled above through the public setters.
        if key.startswith("mlflow."):
            continue
        reduced = _cap(_summarize_bulky(value) if key in _BULKY_KEYS else summarize_payload(value))
        if reduced != value:
            span.set_attribute(key, reduced)


def install_payload_guard() -> None:
    """Register ``payload_guard`` as a global MLflow span processor."""
    import mlflow

    mlflow.tracing.configure(span_processors=[payload_guard])


def annotate_active_span(**attributes: Any) -> None:
    """Set ``attributes`` on the currently active MLflow span, if there is one.

    No-op when MLflow was never imported (tracing disabled) or no span is
    active, so callers outside the tracing path don't pay for an import.
    """
    mlflow = sys.modules.get("mlflow")
    if mlflow is None:
        return
    span = mlflow.get_current_active_span()
    if span is None:
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


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
    the same id. Re-entering a run that's already active in the current
    context raises ("Run ... is already active"), so this checks first and
    no-ops if so.

    Must *not* terminate the run on exit: the run is owned by the thread that
    started it and is still in progress. ``with mlflow.start_run(run_id=...)``
    calls ``end_run()`` when the block exits, which marks the run FINISHED on
    the tracking server as soon as the first worker task returns (verified
    empirically) -- so this attaches the run to the calling thread's
    active-run stack directly instead (see ``_attach_run``).
    """
    if not run_id:
        yield
        return
    import mlflow

    active = mlflow.active_run()
    if active is not None and active.info.run_id == run_id:
        yield
        return

    with _attach_run(run_id):
        yield


_run_cache: dict[str, Any] = {}
_run_cache_lock = threading.Lock()


def _cached_run(run_id: str) -> Any:
    """Fetch (once per process) the ``Run`` entity for ``run_id``.

    Only ``run.info`` (id, experiment) is used by the active-run stack, which
    doesn't change over the run's lifetime, so a stale ``run.data`` is fine.
    """
    with _run_cache_lock:
        run = _run_cache.get(run_id)
    if run is None:
        from mlflow import MlflowClient

        run = MlflowClient().get_run(run_id)
        with _run_cache_lock:
            _run_cache.setdefault(run_id, run)
    return run


@contextmanager
def _attach_run(run_id: str) -> Generator[None, None, None]:
    """Make ``run_id`` the active run for the current thread without ending it
    on exit.

    Pushes onto / pops from MLflow's thread-local active-run stack. That's an
    internal API (``mlflow.tracking.fluent._active_run_stack``), not part of
    MLflow's public surface, so this degrades gracefully if a future MLflow
    version removes or renames it: it falls back to
    ``mlflow.start_run(run_id=...)`` and restores RUNNING status afterwards
    (racy, but keeps traces attached to the right run).
    """
    try:
        from mlflow.tracking.fluent import ActiveRun, _active_run_stack

        stack = _active_run_stack.get()
    except (ImportError, AttributeError):
        stack = None

    if stack is None:
        import mlflow

        with mlflow.start_run(run_id=run_id):
            yield
        mlflow.MlflowClient().update_run(run_id, status="RUNNING")
        return

    entry = ActiveRun(_cached_run(run_id))
    stack.append(entry)
    try:
        yield
    finally:
        # Remove exactly our entry, even if something nested pushed/popped.
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] is entry:
                del stack[i]
                break


@contextmanager
def _no_ambient_span() -> Generator[None, None, None]:
    """Suppress whatever span MLflow considers "currently active" for the
    duration of this block, so a new root span created inside it is a true,
    independent root rather than implicitly nesting under it.

    Needed because ``mlflow.start_span_no_context(name)`` with no explicit
    ``parent_span`` does *not* ignore ambient context -- verified empirically
    that it silently attaches to whatever span is active (e.g. one created by
    an instrumentation library's own span around the LangGraph task that's
    calling this). When multiple documents' tasks share that ambient
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
    "summarize_payload",
    "payload_guard",
    "install_payload_guard",
    "annotate_active_span",
    "session_context",
    "run_context",
    "start_document_span",
    "start_child_span",
    "end_span",
]
