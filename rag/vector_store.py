"""
src/rag/vector_store.py — Qdrant vector store operations.

Spec §1: hard namespace isolation → one Qdrant collection per tenant+namespace.
Collection naming: {tenant_id}__{namespace_id}  (double underscore separator)

Hybrid search: new collections are created with both a named dense vector
("dense") and a named sparse vector ("sparse"). Older collections that have
only an unnamed dense vector are auto-detected and fall back to dense-only
search. No manual migration needed.

Operations:
  - ensure_collection   → create if not exists (hybrid format)
  - upsert_chunks       → store dense + sparse vectors
  - search_chunks       → hybrid RRF search or dense-only fallback
  - delete_source       → remove all chunks for a source_id
  - delete_namespace    → drop entire collection (GDPR)
  - get_stats           → chunk count, source count, last indexed
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from rag.models import Chunk, NamespaceStats

load_dotenv()

log = logging.getLogger(__name__)

# ── Supported embedding models (proxy-confirmed) ───────────────────────────────

_SUPPORTED_MODELS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

QDRANT_URL      = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

if EMBEDDING_MODEL not in _SUPPORTED_MODELS:
    raise RuntimeError(
        f"Unsupported EMBEDDING_MODEL='{EMBEDDING_MODEL}'. "
        f"Supported: {list(_SUPPORTED_MODELS)}"
    )

EMBEDDING_DIM = _SUPPORTED_MODELS[EMBEDDING_MODEL]

# ── Helpers ────────────────────────────────────────────────────────────────────

def collection_name(tenant_id: str, namespace_id: str) -> str:
    """Spec §1: hard isolation — one collection per tenant+namespace."""
    return f"{tenant_id}__{namespace_id}"


async def _is_hybrid(client: AsyncQdrantClient, col: str) -> bool:
    """Return True if this collection was created with sparse vector support."""
    try:
        info = await client.get_collection(col)
        return bool(info.config.params.sparse_vectors_config)
    except Exception:
        return False


async def _has_named_dense(client: AsyncQdrantClient, col: str) -> bool:
    """Return True if this collection uses named vectors (not legacy unnamed dense)."""
    try:
        info = await client.get_collection(col)
        return isinstance(info.config.params.vectors, dict)
    except Exception:
        return False


# ── Collection management ──────────────────────────────────────────────────────

async def ensure_collection(client: AsyncQdrantClient, name: str) -> None:
    """Create Qdrant collection (hybrid: dense + sparse) if it does not exist.

    New collections use named vectors:
      "dense"  — VectorParams for cosine similarity (OpenAI embeddings)
      "sparse" — SparseVectorParams for hybrid/BM25-style retrieval

    Existing collections are left untouched (already-exists exception is caught).
    """
    try:
        await client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        log.info("Created Qdrant collection", extra={"collection": name, "mode": "hybrid"})
    except Exception as exc:
        if "already exists" not in str(exc).lower():
            raise


# ── Write operations ───────────────────────────────────────────────────────────

async def upsert_chunks(
    client: AsyncQdrantClient,
    col: str,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector] | None = None,
) -> None:
    """Store chunks with dense (and optionally sparse) vectors in Qdrant.

    If sparse_vectors is provided and the collection supports sparse, both
    vectors are stored (hybrid mode). Otherwise only dense is stored.
    """
    hybrid = sparse_vectors is not None and await _is_hybrid(client, col)
    named = hybrid or await _has_named_dense(client, col)

    points: list[PointStruct] = []
    for i, (chunk, dense) in enumerate(zip(chunks, dense_vectors)):
        if hybrid and sparse_vectors is not None:
            vector: dict | list = {
                "dense":  dense,
                "sparse": sparse_vectors[i],
            }
        elif named:
            vector = {"dense": dense}  # named dense only (no sparse)
        else:
            vector = dense  # legacy unnamed dense — works with old collections

        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector=vector,
                payload={
                    "chunk_id":     chunk.chunk_id,
                    "content":      chunk.content,
                    "source_id":    chunk.source_id,
                    "namespace_id": chunk.namespace_id,
                    "page_number":  chunk.page_number,
                    "metadata":     chunk.metadata or {},
                    "indexed_at":   datetime.now(timezone.utc).isoformat(),
                },
            )
        )

    await client.upsert(collection_name=col, points=points)
    log.info(
        "Upserted chunks",
        extra={"collection": col, "count": len(points), "hybrid": hybrid},
    )


# ── Read operations ────────────────────────────────────────────────────────────

async def search_chunks(
    client: AsyncQdrantClient,
    col: str,
    dense_vector: list[float],
    top_k: int,
    hint_page_number: int | None = None,
    sparse_vector: SparseVector | None = None,
) -> list[Chunk]:
    """Semantic search — hybrid RRF (if collection supports it) or dense-only.

    Hybrid mode: Qdrant RRF fusion of dense cosine + sparse TF scores.
    Dense-only:  Classic cosine similarity (backward compat with old collections).

    Page boost: if hint_page_number is set, results are filtered to that page
    (in dense-only mode: slot-filling strategy; in hybrid: post-fusion filter).
    """
    # Collection may not exist yet (no successful ingest)
    try:
        existing = {c.name for c in (await client.get_collections()).collections}
        if col not in existing:
            return []
    except Exception:
        return []

    hybrid = sparse_vector is not None and await _is_hybrid(client, col)

    if hybrid:
        return await _search_hybrid(client, col, dense_vector, sparse_vector, top_k, hint_page_number)
    else:
        return await _search_dense(client, col, dense_vector, top_k, hint_page_number)


async def _search_hybrid(
    client: AsyncQdrantClient,
    col: str,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    top_k: int,
    hint_page_number: int | None,
) -> list[Chunk]:
    """Hybrid search using Qdrant RRF fusion of dense + sparse prefetch."""
    page_filter = (
        Filter(must=[FieldCondition(key="page_number", match=MatchValue(value=hint_page_number))])
        if hint_page_number else None
    )

    hits = (await client.query_points(
        collection_name=col,
        prefetch=[
            Prefetch(query=dense_vector,  using="dense",  limit=top_k * 3),
            Prefetch(query=sparse_vector, using="sparse", limit=top_k * 3),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=page_filter,
        limit=top_k,
        with_payload=True,
    )).points

    return [_hit_to_chunk(h) for h in hits]


async def _search_dense(
    client: AsyncQdrantClient,
    col: str,
    query_vector: list[float],
    top_k: int,
    hint_page_number: int | None,
) -> list[Chunk]:
    """Dense-only cosine search (legacy collections without sparse vectors).

    Uses slot-filling: page-filtered results first, then unfiltered fill.
    """
    results: list[Chunk] = []
    seen_ids: set = set()

    # Step A: page-boosted results first
    if hint_page_number:
        page_filter = Filter(
            must=[FieldCondition(key="page_number", match=MatchValue(value=hint_page_number))]
        )
        hits = (await client.query_points(
            collection_name=col,
            query=query_vector,
            using="dense",
            query_filter=page_filter,
            limit=top_k,
            with_payload=True,
        )).points
        for h in hits:
            results.append(_hit_to_chunk(h))
            seen_ids.add(h.id)

    # Step B: fill remaining slots with unfiltered search
    remaining = top_k - len(results)
    if remaining > 0:
        hits = (await client.query_points(
            collection_name=col,
            query=query_vector,
            using="dense",
            limit=top_k,
            with_payload=True,
        )).points
        for h in hits:
            if h.id not in seen_ids and len(results) < top_k:
                results.append(_hit_to_chunk(h))
                seen_ids.add(h.id)

    return results


def _hit_to_chunk(hit) -> Chunk:
    """Convert a Qdrant ScoredPoint to a Chunk model."""
    p = hit.payload
    return Chunk(
        chunk_id=p["chunk_id"],
        content=p["content"],
        source_id=p["source_id"],
        namespace_id=p["namespace_id"],
        page_number=p.get("page_number"),
        score=round(max(0.0, min(1.0, hit.score)), 4),
        metadata=p.get("metadata", {}),
    )


# ── Delete operations ──────────────────────────────────────────────────────────

async def delete_source(
    client: AsyncQdrantClient,
    col: str,
    source_id: str,
) -> None:
    """Delete all chunks belonging to source_id.

    Spec §6: DELETE /v1/namespaces/{namespace_id}/sources/{source_id}
    """
    await client.delete(
        collection_name=col,
        points_selector=Filter(
            must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
        ),
    )


async def delete_namespace(client: AsyncQdrantClient, col: str) -> None:
    """Drop entire collection — removes all chunks for the namespace.

    Spec §7: DELETE /v1/namespaces/{namespace_id}  (GDPR hard delete)
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if col in existing:
        await client.delete_collection(col)


# ── Stats ──────────────────────────────────────────────────────────────────────

async def get_stats(
    client: AsyncQdrantClient,
    col: str,
    namespace_id: str,
) -> NamespaceStats | None:
    """Return chunk count, source count, last indexed timestamp.

    Spec §5: GET /v1/namespaces/{namespace_id}/stats
    Returns None when the collection does not exist (caller raises 404).
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if col not in existing:
        return None

    info = await client.get_collection(col)
    chunk_count = info.points_count or 0

    # Scroll all payloads to count distinct sources and find latest timestamp
    source_ids: set[str] = set()
    last_indexed: str | None = None
    offset = None

    while True:
        records, offset = await client.scroll(
            collection_name=col,
            with_payload=True,
            limit=256,
            offset=offset,
        )
        for r in records:
            if r.payload:
                source_ids.add(r.payload.get("source_id", ""))
                ts = r.payload.get("indexed_at")
                if ts and (last_indexed is None or ts > last_indexed):
                    last_indexed = ts
        if offset is None:
            break

    return NamespaceStats(
        namespace_id=namespace_id,
        chunk_count=chunk_count,
        source_count=len(source_ids),
        total_tokens_indexed=0,
        last_ingested_at=last_indexed,
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=EMBEDDING_DIM,
    )


# ── Stats cache (Redis-backed) ────────────────────────────────────────────────

_STATS_CACHE_PREFIX = "rag:stats:"


async def get_stats_cached(
    client: AsyncQdrantClient,
    col: str,
    namespace_id: str,
    redis=None,
) -> NamespaceStats | None:
    """Like get_stats() but caches the result in Redis for STATS_CACHE_TTL_S seconds.

    Pass redis=None to skip caching (e.g. in tests).
    Cache is invalidated by invalidate_stats_cache() after every ingest.
    """
    from rag.config import STATS_CACHE_TTL_S  # local import avoids circular dep

    if redis is None or STATS_CACHE_TTL_S == 0:
        return await get_stats(client, col, namespace_id)

    cache_key = f"{_STATS_CACHE_PREFIX}{col}"
    try:
        raw = await redis.get(cache_key)
        if raw:
            return NamespaceStats(**json.loads(raw))
    except Exception:
        pass

    stats = await get_stats(client, col, namespace_id)
    if stats is not None:
        try:
            await redis.set(cache_key, stats.model_dump_json(), ex=STATS_CACHE_TTL_S)
        except Exception:
            pass
    return stats


async def invalidate_stats_cache(redis, col: str) -> None:
    """Delete the cached stats entry for a collection.

    Called from run_ingest() after every successful ingest so /stats always
    reflects the current chunk and source counts.
    """
    if redis is None:
        return
    try:
        await redis.delete(f"{_STATS_CACHE_PREFIX}{col}")
    except Exception:
        pass
