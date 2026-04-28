"""
rag/worker.py — Standalone ingest worker process.

Pulls jobs from the Redis queue (rag:queue:ingest) and runs the full
ingest pipeline. Designed to run alongside the FastAPI service:

    python rag/worker.py

Or via Docker Compose (worker service in compose.yml).

Environment variables: identical to rag_server.py.
Concurrency: WORKER_CONCURRENCY parallel asyncio tasks (default: 4).

Graceful shutdown: SIGINT / SIGTERM → drains in-flight jobs → exits.
"""

import asyncio
import logging
import math
import os
import signal
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient

from rag.logging_config import setup_logging
from rag.models import IngestRequest
from rag.queue import dequeue_ingest, enqueue_retry, move_to_dlq
from rag.ingest import run_ingest
from rag.metrics import INGEST_RETRIES

log = logging.getLogger("rag.worker")

QDRANT_URL   = os.environ.get("QDRANT_URL",              "http://localhost:6333")
REDIS_URL    = os.environ.get("REDIS_URL",               "redis://localhost:6379")
CONCURRENCY  = int(os.environ.get("WORKER_CONCURRENCY",  "4"))
MAX_RETRIES  = int(os.environ.get("WORKER_MAX_RETRIES",  "3"))
RETRY_DELAY  = float(os.environ.get("WORKER_RETRY_DELAY_S", "5"))

# ── Error classification ───────────────────────────────────────────────────────

#: Exception types that should NOT be retried — permanent failures.
#: Network errors (httpx), Qdrant/Redis connection errors → retryable (default).
_NON_RETRYABLE_TYPES = (ValueError, NotImplementedError, TypeError)

#: Additional non-retryable substrings in the error message (lowercase).
_NON_RETRYABLE_MSGS = (
    "empty content",
    "unsupported mime",
    "unsupported format",
    "validation",
)


def _is_retryable(exc: Exception) -> bool:
    """Return False for permanent failures that should bypass retry and go to DLQ."""
    if isinstance(exc, _NON_RETRYABLE_TYPES):
        return False
    msg = str(exc).lower()
    return not any(kw in msg for kw in _NON_RETRYABLE_MSGS)

_shutdown = asyncio.Event()


def _handle_signal(sig, frame) -> None:  # noqa: ANN001
    log.info("Shutdown signal received", extra={"signal": sig})
    _shutdown.set()


async def _process_one(redis: aioredis.Redis, qdrant: AsyncQdrantClient) -> bool:
    """Dequeue and process one job. Returns True if a job was processed."""
    job = await dequeue_ingest(redis, timeout=5)
    if job is None:
        return False

    # Honour backoff window set by enqueue_retry
    retry_not_before = job.get("retry_not_before", 0.0)
    if time.time() < retry_not_before:
        # Not yet — put it back and yield to other workers
        await redis.rpush("rag:queue:ingest", __import__("json").dumps(job))
        await asyncio.sleep(0.5)
        return False

    job_id       = job["job_id"]
    tenant_id    = job["tenant_id"]
    namespace_id = job["namespace_id"]
    source_id    = job["source_id"]
    req          = IngestRequest.model_validate(job["req"])
    file_hex     = job.get("file_bytes_hex")
    file_bytes   = bytes.fromhex(file_hex) if file_hex else None
    retry_count  = job.get("retry_count", 0)

    log.info(
        "Processing ingest job",
        extra={"job_id": job_id, "source_id": source_id, "tenant_id": tenant_id, "retry_count": retry_count},
    )
    try:
        await run_ingest(
            r=redis,
            qdrant=qdrant,
            job_id=job_id,
            tenant_id=tenant_id,
            namespace_id=namespace_id,
            source_id=source_id,
            req=req,
            file_bytes=file_bytes,
        )
        log.info("Ingest job completed", extra={"job_id": job_id})
    except Exception as exc:
        log.error(
            "Ingest job failed",
            extra={"job_id": job_id, "error": str(exc), "retry_count": retry_count},
            exc_info=True,
        )
        next_retry = retry_count + 1
        if next_retry <= MAX_RETRIES and _is_retryable(exc):
            INGEST_RETRIES.inc()
            delay = RETRY_DELAY * math.pow(2, retry_count)  # exponential backoff
            await enqueue_retry(redis, job, retry_count=next_retry, delay_s=delay)
        else:
            if not _is_retryable(exc):
                log.warning(
                    "Non-retryable error — moving directly to DLQ",
                    extra={"job_id": job_id, "error": str(exc)},
                )
            await move_to_dlq(redis, job)
    return True


async def _worker_loop(worker_id: int, redis: aioredis.Redis, qdrant: AsyncQdrantClient) -> None:
    """Single worker coroutine — runs until _shutdown is set."""
    log.info("Worker started", extra={"worker_id": worker_id})
    while not _shutdown.is_set():
        try:
            await _process_one(redis, qdrant)
        except Exception as exc:
            log.error(
                "Worker loop error — retrying in 1s",
                extra={"worker_id": worker_id, "error": str(exc)},
            )
            await asyncio.sleep(1)
    log.info("Worker stopped", extra={"worker_id": worker_id})


async def run_workers(
    qdrant: AsyncQdrantClient,
    redis: aioredis.Redis,
    concurrency: int = CONCURRENCY,
) -> None:
    """Run ingest worker loops until cancelled.

    Designed to be embedded in the FastAPI server lifespan as a background task:

        task = asyncio.create_task(run_workers(qdrant, redis))
        yield
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    """
    log.info("Starting embedded ingest workers", extra={"concurrency": concurrency})
    tasks = [
        asyncio.create_task(_worker_loop(i, redis, qdrant))
        for i in range(concurrency)
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("Embedded workers shut down cleanly")
        raise


async def main() -> None:
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    log.info("Starting RAG ingest workers", extra={"concurrency": CONCURRENCY})

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    redis  = await aioredis.from_url(REDIS_URL, decode_responses=False)
    qdrant = AsyncQdrantClient(url=QDRANT_URL)

    tasks = [
        asyncio.create_task(_worker_loop(i, redis, qdrant))
        for i in range(CONCURRENCY)
    ]

    await _shutdown.wait()

    log.info("Draining in-flight jobs…")
    await asyncio.gather(*tasks, return_exceptions=True)

    await qdrant.close()
    await redis.aclose()
    log.info("Workers shut down cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
