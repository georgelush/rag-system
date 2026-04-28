"""
rag/cache.py — Redis-backed query result cache.

Cache key:  rag:qcache:{tenant_id}:{sha256(question + sorted_namespaces + top_k)}
Invalidation key: rag:qcache:ns:{tenant_id}:{namespace_id}  → set of cache keys to delete

Architecture
------------
- On cache hit  → return stored QueryResponse JSON; skip embed + search + LLM.
- On cache miss → run full pipeline, store result, track key in per-namespace invalidation set.
- On ingest     → call invalidate_namespace() to delete all cached keys for that namespace.

Env vars
--------
  QUERY_CACHE_ENABLED   (default "true")
  QUERY_CACHE_TTL_S     (default "300")  — 5 minutes
"""

import hashlib
import json
import logging
import os

from redis.asyncio import Redis

from rag.metrics import CACHE_HITS, CACHE_MISSES

log = logging.getLogger(__name__)

CACHE_ENABLED = os.environ.get("QUERY_CACHE_ENABLED", "true").lower() != "false"
CACHE_TTL_S   = int(os.environ.get("QUERY_CACHE_TTL_S", "300"))

_CACHE_PREFIX = "rag:qcache:"
_NS_INDEX_PREFIX = "rag:qcache:ns:"


def _cache_key(tenant_id: str, question: str, namespaces: list[str], top_k: int) -> str:
    """Deterministic cache key — stable regardless of namespace order."""
    fingerprint = json.dumps(
        {
            "q": question,
            "ns": sorted(namespaces),
            "k": top_k,
        },
        separators=(",", ":"),
    )
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:32]
    return f"{_CACHE_PREFIX}{tenant_id}:{digest}"


def _ns_index_key(tenant_id: str, namespace_id: str) -> str:
    return f"{_NS_INDEX_PREFIX}{tenant_id}:{namespace_id}"


async def get_cached(
    redis: Redis,
    tenant_id: str,
    question: str,
    namespaces: list[str],
    top_k: int,
) -> dict | None:
    """Return cached QueryResponse dict, or None on miss/disabled."""
    if not CACHE_ENABLED:
        return None
    key = _cache_key(tenant_id, question, namespaces, top_k)
    try:
        raw = await redis.get(key)
        if raw:
            CACHE_HITS.inc()
            log.info(
                "Query cache hit",
                extra={"tenant_id": tenant_id, "cache_key": key},
            )
            return json.loads(raw)
        CACHE_MISSES.inc()
    except Exception as exc:
        log.warning("Cache get failed", extra={"error": str(exc)})
    return None


async def set_cached(
    redis: Redis,
    tenant_id: str,
    question: str,
    namespaces: list[str],
    top_k: int,
    response_dict: dict,
) -> None:
    """Store QueryResponse dict in cache and register key in per-namespace indexes."""
    if not CACHE_ENABLED:
        return
    key = _cache_key(tenant_id, question, namespaces, top_k)
    try:
        await redis.set(key, json.dumps(response_dict), ex=CACHE_TTL_S)
        # Register key in every namespace's invalidation set so we can bulk-delete on re-ingest
        for ns_id in namespaces:
            idx = _ns_index_key(tenant_id, ns_id)
            await redis.sadd(idx, key)
            await redis.expire(idx, CACHE_TTL_S * 2)  # index lives twice as long
        log.info(
            "Query cached",
            extra={"tenant_id": tenant_id, "cache_key": key, "ttl_s": CACHE_TTL_S},
        )
    except Exception as exc:
        log.warning("Cache set failed", extra={"error": str(exc)})


async def invalidate_namespace(
    redis: Redis,
    tenant_id: str,
    namespace_id: str,
) -> int:
    """Delete all cached query results that touch `namespace_id`. Returns count deleted."""
    if not CACHE_ENABLED:
        return 0
    idx = _ns_index_key(tenant_id, namespace_id)
    try:
        keys = await redis.smembers(idx)
        if keys:
            await redis.delete(*keys, idx)
            log.info(
                "Cache invalidated for namespace",
                extra={
                    "tenant_id":    tenant_id,
                    "namespace_id": namespace_id,
                    "keys_deleted": len(keys),
                },
            )
            return len(keys)
        await redis.delete(idx)
    except Exception as exc:
        log.warning("Cache invalidation failed", extra={"error": str(exc)})
    return 0
