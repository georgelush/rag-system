"""
rag/ingest.py — Document ingest pipeline.

Stages:
  fetching    → download URL or read uploaded file bytes
  extracting  → detect format, extract plain text (extractor.py)
  chunking    → split text using pluggable chunker (chunkers/)
  embedding   → call embeddings API (batched) + sparse encoding
  indexing    → upsert dense + sparse vectors into Qdrant
  done / failed
"""

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone

import httpx
from redis.asyncio import Redis
from qdrant_client import AsyncQdrantClient

from rag.extractor import extract
from rag.chunkers import get_chunker, PageChunker
from rag.jobs import update_job
from rag.language import detect_language
from rag.models import Chunk, IngestRequest, IngestStatus
from rag.openai_client import openai_client
from rag.sparse_encoder import encode as encode_sparse
from rag.config import EMBED_BATCH_SIZE
from rag.vector_store import (
    EMBEDDING_MODEL,
    collection_name,
    ensure_collection,
    delete_source,
    upsert_chunks,
)
from rag import telemetry

log = logging.getLogger(__name__)

WEBHOOK_SECRET: str = os.environ.get("WEBHOOK_SECRET", "")


async def fetch_content(req: IngestRequest, file_bytes: bytes | None = None) -> tuple[bytes, str]:
    """Stage: fetching — download URL or return uploaded file bytes.

    Returns:
        (raw_bytes, filename) — filename used for MIME detection in extractor.
    """
    if req.source_type.value == "url":
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(str(req.url))
            response.raise_for_status()
            # Extract filename from URL path for MIME detection
            filename = str(req.url).split("/")[-1].split("?")[0]
            return response.content, filename
    else:
        return file_bytes or b"", ""


async def _send_callback(callback_url: str | None, payload: dict) -> bool:
    """POST callback_url with HMAC-SHA256 signature when job completes/fails.

    Returns True if the callback was delivered (HTTP 2xx), False otherwise.
    Does nothing (returns False) when callback_url is None.
    Retries up to 3 times with exponential backoff.
    """
    if not callback_url:
        return False

    body = json.dumps(payload, separators=(",", ":")).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}

    if WEBHOOK_SECRET:
        sig = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
        headers["X-Vendor-Signature"] = f"sha256={sig}"

    import asyncio
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(3):
            try:
                resp = await client.post(callback_url, content=body, headers=headers)
                if resp.status_code < 300:
                    return True
            except Exception:
                pass
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)  # 1 s, 2 s

    log.warning("Callback delivery failed after 3 attempts", extra={"url": callback_url})
    return False


async def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Stage: embedding — call OpenAI embeddings API for all chunks.

    Batches up to 100 chunks per API call to stay within rate limits.
    Returns dense vectors only. Sparse vectors are generated separately
    by encode_sparse_chunks() without any API call.
    """
    vectors: list[list[float]] = []
    batch_size = EMBED_BATCH_SIZE

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[c.content for c in batch],
        )
        vectors.extend([item.embedding for item in response.data])

    return vectors


def encode_sparse_chunks(chunks: list[Chunk]):
    """Generate sparse TF vectors for all chunks (no API call, local only)."""
    from qdrant_client.models import SparseVector
    return [encode_sparse(c.content) for c in chunks]


async def run_ingest(
    r: Redis,
    qdrant: AsyncQdrantClient,
    job_id: str,
    tenant_id: str,
    namespace_id: str,
    source_id: str,
    req: IngestRequest,
    file_bytes: bytes | None = None,
) -> None:
    """Orchestrate the full ingest pipeline. Runs as background task or worker job.

    Updates job status in Redis at each stage so caller can poll
    GET /v1/ingest/{job_id}.
    """
    col = collection_name(tenant_id, namespace_id)

    # Start Langfuse trace for this ingest job
    trace = telemetry.start_trace(
        "ingest",
        trace_id=job_id,
        user_id=tenant_id,
        metadata={
            "source_id":    source_id,
            "namespace_id": namespace_id,
            "source_type":  req.source_type.value,
            "strategy":     req.chunking.strategy.value,
        },
    )

    try:
        # Stage 1: fetching
        await update_job(r, job_id, IngestStatus.fetching, 10, 0)
        span = trace.span(name="fetch", input={"url": str(req.url) if req.url else "file"})
        raw, filename = await fetch_content(req, file_bytes=file_bytes)
        span.end(output={"bytes": len(raw), "filename": filename})

        # Stage 2: extracting
        await update_job(r, job_id, IngestStatus.extracting, 30, 0)
        span = trace.span(name="extract", input={"filename": filename})
        result = await extract(raw, filename=filename, mime_hint=req.mime_type_hint or "")
        if not result.text.strip():
            raise ValueError("Empty content after extraction.")
        language: str = req.language or detect_language(result.text)
        span.end(output={"chars": len(result.text), "pages": len(result.pages or []), "language": language, "mime": result.mime_type})

        # Stage 3: chunking
        await update_job(r, job_id, IngestStatus.chunking, 50, 0)
        span = trace.span(name="chunk", input={"strategy": req.chunking.strategy.value})
        chunker = get_chunker(req.chunking, text=result.text)
        if isinstance(chunker, PageChunker) and result.pages:
            chunks = chunker.chunk_pages(result.pages, source_id, namespace_id, language)
        else:
            chunks = chunker.chunk(result.text, source_id, namespace_id, language)
        span.end(output={"chunk_count": len(chunks)})

        # Stage 4: embedding — dense (API) + sparse (local, no API call)
        await update_job(r, job_id, IngestStatus.embedding, 70, 0)
        span = trace.span(name="embed", input={"chunk_count": len(chunks), "model": EMBEDDING_MODEL})
        dense_vectors  = await embed_chunks(chunks)
        sparse_vectors = encode_sparse_chunks(chunks)
        span.end(output={"dense_dims": len(dense_vectors[0]) if dense_vectors else 0})

        # Stage 5: indexing — ensure collection, deduplicate, upsert
        await update_job(r, job_id, IngestStatus.indexing, 90, 0)
        span = trace.span(name="index", input={"collection": col, "chunk_count": len(chunks)})
        await ensure_collection(qdrant, col)
        if not req.incremental:
            await delete_source(qdrant, col, source_id)  # remove stale chunks
        await upsert_chunks(qdrant, col, chunks, dense_vectors, sparse_vectors)
        span.end(output={"indexed": len(chunks)})

        # Done — invalidate any cached query results and stats cache
        from rag.cache import invalidate_namespace
        from rag.vector_store import invalidate_stats_cache
        from rag.jobs import update_callback_status
        await invalidate_namespace(r, tenant_id, namespace_id)
        await invalidate_stats_cache(r, col)

        await update_job(r, job_id, IngestStatus.done, 100, len(chunks))
        log.info(
            "Ingest completed",
            extra={
                "job_id":       job_id,
                "source_id":    source_id,
                "tenant_id":    tenant_id,
                "chunks":       len(chunks),
                "language":     language,
            },
        )
        cb_ok = await _send_callback(req.callback_url, {
            "event":         "ingest.completed",
            "job_id":        job_id,
            "namespace_id":  namespace_id,
            "source_id":     source_id,
            "status":        "done",
            "chunks_created": len(chunks),
            "at":            datetime.now(timezone.utc).isoformat(),
        })
        if req.callback_url:
            await update_callback_status(r, job_id, "sent" if cb_ok else "failed")

    except Exception as exc:
        from rag.models import IngestError, ErrorCode
        from rag.jobs import update_callback_status
        _retryable = not isinstance(exc, (ValueError, NotImplementedError))
        await update_job(
            r, job_id, IngestStatus.failed, 0, 0,
            error=IngestError(
                code=ErrorCode.internal_error,
                message=str(exc),
                retryable=_retryable,
            ),
        )
        cb_ok = await _send_callback(req.callback_url, {
            "event": "ingest.failed",
            "job_id": job_id,
            "namespace_id": namespace_id,
            "source_id": source_id,
            "status": "failed",
            "chunks_created": 0,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        if req.callback_url:
            await update_callback_status(r, job_id, "sent" if cb_ok else "failed")