"""
rag/telemetry.py — Langfuse observability client (optional).

All public functions are no-ops when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
are not set. Safe to import and call anywhere without try/except.

Trace structure:
  Query  → Trace("query")
               └─ Span("embed_question")
               └─ Span("search_chunks")
               └─ Generation("generate_answer", model=..., usage=...)
  Ingest → Trace("ingest")
               └─ Span("fetch")
               └─ Span("extract")
               └─ Span("chunk")
               └─ Span("embed")
               └─ Span("index")
"""

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
_HOST       = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

_lf = None  # singleton Langfuse client — lazy-initialised on first call


class _Noop:
    """Null object — absorbs all method calls silently when Langfuse is disabled."""

    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any) -> "_Noop":
            return self
        return _noop

    def __enter__(self) -> "_Noop":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _client():
    global _lf
    if _lf is not None:
        return _lf
    if not _PUBLIC_KEY or not _SECRET_KEY:
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import-untyped]
        _lf = Langfuse(
            public_key=_PUBLIC_KEY,
            secret_key=_SECRET_KEY,
            host=_HOST,
            debug=False,
        )
        log.info("Langfuse client ready", extra={"host": _HOST})
    except Exception as exc:
        log.warning("Langfuse unavailable — tracing disabled", extra={"error": str(exc)})
        _lf = None
    return _lf


def start_trace(
    name: str,
    *,
    trace_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Start a Langfuse trace.

    Returns a StatefulTraceClient (Langfuse) or _Noop() when disabled.
    The returned object supports .span(), .generation(), .update().
    """
    lf = _client()
    if lf is None:
        return _Noop()
    try:
        kwargs: dict[str, Any] = {"name": name}
        if trace_id:
            kwargs["id"] = trace_id
        if user_id:
            kwargs["user_id"] = user_id
        if session_id:
            kwargs["session_id"] = session_id
        if metadata:
            kwargs["metadata"] = metadata
        return lf.trace(**kwargs)
    except Exception as exc:
        log.debug("Langfuse trace creation failed", extra={"error": str(exc)})
        return _Noop()


def flush() -> None:
    """Flush all pending Langfuse events. Call on server shutdown."""
    lf = _client()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass
