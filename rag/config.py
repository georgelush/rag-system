"""rag/config.py — Centralised environment-variable configuration.

All new configurable values introduced in Session D onward live here.
Existing module-level os.environ.get() calls in other files are kept for
backward compatibility; this file only adds *new* knobs.

Usage
-----
    from rag.config import EMBED_BATCH_SIZE, RERANKER_TIMEOUT_S, ...
"""

import os


# ── Required ───────────────────────────────────────────────────────────────────

RAG_API_KEY: str = os.environ.get("RAG_API_KEY", "")


# ── Embedding ──────────────────────────────────────────────────────────────────

#: Chunks per OpenAI embeddings.create() call.
#: OpenAI supports up to 2,048; default raised from legacy 100 → 500.
EMBED_BATCH_SIZE: int = int(os.environ.get("EMBED_BATCH_SIZE", "500"))


# ── Reranker ───────────────────────────────────────────────────────────────────

RERANKER_URL: str = os.environ.get("RERANKER_URL", "")

#: HTTP timeout for calls to the cross-encoder reranker sidecar.
RERANKER_TIMEOUT_S: float = float(os.environ.get("RERANKER_TIMEOUT_S", "10.0"))


# ── Stats cache ────────────────────────────────────────────────────────────────

#: TTL (seconds) for GET /v1/namespaces/{id}/stats Redis cache.
#: Set to 0 to disable caching (every request scrolls Qdrant directly).
STATS_CACHE_TTL_S: int = int(os.environ.get("STATS_CACHE_TTL_S", "60"))


# ── Worker ─────────────────────────────────────────────────────────────────────

#: Max seconds to wait for in-flight jobs during graceful shutdown.
WORKER_SHUTDOWN_TIMEOUT_S: int = int(os.environ.get("WORKER_SHUTDOWN_TIMEOUT_S", "30"))


# ── Batch query ────────────────────────────────────────────────────────────────

#: Maximum number of parallel queries accepted in POST /v1/query/batch.
BATCH_QUERY_MAX: int = int(os.environ.get("BATCH_QUERY_MAX", "10"))


# ── Startup validation ─────────────────────────────────────────────────────────

def validate_startup() -> list[str]:
    """Return a list of human-readable warning strings for missing/bad config.

    Call once at lifespan startup — log each entry at WARNING level.
    Raises RuntimeError immediately if RAG_API_KEY is absent.
    """
    if not RAG_API_KEY:
        raise RuntimeError(
            "RAG_API_KEY environment variable is required but not set."
        )

    warnings: list[str] = []

    if not os.environ.get("WEBHOOK_SECRET"):
        warnings.append(
            "WEBHOOK_SECRET is not set — webhook callback signatures will be omitted "
            "(callers cannot verify delivery authenticity)."
        )

    rbac_enabled = os.environ.get("RBAC_ENABLED", "false").lower() == "true"
    rbac_secret = os.environ.get("RBAC_SECRET", "")
    if rbac_enabled and not rbac_secret:
        warnings.append(
            "RBAC_ENABLED=true but RBAC_SECRET is not set — "
            "POST /v1/tokens will return 500 at runtime."
        )

    pub = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
    sec = bool(os.environ.get("LANGFUSE_SECRET_KEY"))
    if pub != sec:
        warnings.append(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must both be set or both absent "
            "(only one is currently set — tracing will be disabled)."
        )

    return warnings
