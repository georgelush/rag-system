"""
rag/metrics.py — Prometheus instrumentation.

Exposes:
  - HTTP request counter + latency histogram (per method/path/status)
  - Ingest queue depth gauge (sampled at scrape time)
  - DLQ depth gauge (sampled at scrape time)

Usage
-----
Mount GET /metrics in rag_server.py:

    from rag.metrics import MetricsMiddleware, metrics_response
    app.add_middleware(MetricsMiddleware)
    app.add_route("/metrics", metrics_response)

Env vars
--------
  METRICS_ENABLED  (default "true") — set "false" to disable endpoint entirely.
"""

import time
import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ── New metric definitions (Session D) ────────────────────────────────────────
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from rag.queue import QUEUE_KEY, DLQ_KEY

# ── Metric definitions ─────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=[0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

INGEST_QUEUE_DEPTH = Gauge(
    "rag_ingest_queue_depth",
    "Number of jobs waiting in the ingest queue",
)

DLQ_DEPTH = Gauge(
    "rag_ingest_dlq_depth",
    "Number of jobs in the dead-letter queue",
)

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Query result cache hits (pipeline skipped)",
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Query result cache misses (full pipeline ran)",
)

RERANKER_FAILURES = Counter(
    "rag_reranker_failures_total",
    "Reranker sidecar call failures (fell back to score-sorted order)",
)

INGEST_RETRIES = Counter(
    "rag_ingest_retries_total",
    "Total ingest job retry attempts (before DLQ)",
)

NAMESPACE_QUERIES = Counter(
    "rag_namespace_queries_total",
    "Queries executed per namespace (bounded cardinality — avoid high-volume deployments)",
    ["namespace_id"],
)

# ── Path normalisation ─────────────────────────────────────────────────────────

_PATH_LABEL_MAP = {
    "/v1/query":    "/v1/query",
    "/v1/ingest":   "/v1/ingest",
    "/v1/health":   "/v1/health",
    "/metrics":     "/metrics",
}


def _normalise_path(path: str) -> str:
    """Replace dynamic segments so cardinality stays bounded."""
    if path.startswith("/v1/ingest/"):
        return "/v1/ingest/{job_id}"
    if path.startswith("/v1/namespaces/"):
        parts = path.split("/")
        if len(parts) == 5 and parts[4] == "stats":
            return "/v1/namespaces/{ns}/stats"
        if len(parts) == 6 and parts[4] == "sources":
            return "/v1/namespaces/{ns}/sources/{src}"
        return "/v1/namespaces/{ns}"
    return _PATH_LABEL_MAP.get(path, "other")


# ── Middleware ─────────────────────────────────────────────────────────────────

class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for every HTTP request."""

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        path = _normalise_path(request.url.path)
        REQUEST_COUNT.labels(
            method=request.method,
            path=path,
            status_code=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, path=path).observe(duration)

        return response


# ── Scrape endpoint ────────────────────────────────────────────────────────────

METRICS_ENABLED = os.environ.get("METRICS_ENABLED", "true").lower() != "false"


async def metrics_response(request: Request) -> Response:
    """GET /metrics — Prometheus text-format scrape endpoint.

    Updates queue-depth gauges at scrape time using the Redis client
    stored in app.state.
    """
    if not METRICS_ENABLED:
        return Response(status_code=404)

    redis = getattr(request.app.state, "redis", None)
    if redis is not None:
        try:
            depth = await redis.llen(QUEUE_KEY)
            dlq   = await redis.llen(DLQ_KEY)
            INGEST_QUEUE_DEPTH.set(depth)
            DLQ_DEPTH.set(dlq)
        except Exception:
            pass  # Redis unavailable — keep stale gauge values

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
