"""
rag/queue.py — Redis list-based ingest job queue.

Architecture:
  Producer (ingest_router) → LPUSH rag:queue:ingest  → JSON payload
  Consumer (worker.py)     → BRPOP rag:queue:ingest  → run_ingest()

Each message is a self-contained JSON dict carrying all arguments that
run_ingest() needs. No shared mutable state beyond Redis.

File bytes (multipart uploads) are hex-encoded for JSON transport.
"""

import json
import logging
import time

from redis.asyncio import Redis

log = logging.getLogger(__name__)

QUEUE_KEY = "rag:queue:ingest"
DLQ_KEY   = "rag:queue:dlq"


async def enqueue_ingest(
    redis: Redis,
    *,
    job_id:        str,
    tenant_id:     str,
    namespace_id:  str,
    source_id:     str,
    req_dict:      dict,
    file_bytes_hex: str | None = None,
) -> None:
    """Push an ingest job onto the Redis queue.

    Args:
        file_bytes_hex: raw file bytes encoded as a lowercase hex string,
                        or None for URL-mode ingests.
    """
    payload = json.dumps({
        "job_id":          job_id,
        "tenant_id":       tenant_id,
        "namespace_id":    namespace_id,
        "source_id":       source_id,
        "req":             req_dict,
        "file_bytes_hex":  file_bytes_hex,
    })
    await redis.lpush(QUEUE_KEY, payload)
    log.info(
        "Ingest job enqueued",
        extra={"job_id": job_id, "source_id": source_id, "queue": QUEUE_KEY},
    )


async def dequeue_ingest(redis: Redis, timeout: int = 5) -> dict | None:
    """Block-pop one job from the queue.

    Args:
        timeout: seconds to wait before returning None (no job available).

    Returns:
        Parsed job dict or None if the queue was empty for `timeout` seconds.
    """
    result = await redis.brpop(QUEUE_KEY, timeout=timeout)
    if result is None:
        return None
    _, raw = result
    return json.loads(raw)


async def enqueue_retry(
    redis: Redis,
    payload: dict,
    *,
    retry_count: int,
    delay_s: float,
) -> None:
    """Re-push a failed job to the tail of the ingest queue after a delay.

    The payload is enriched with ``retry_count`` and ``retry_not_before``
    so the consumer can skip it until the backoff window has elapsed.
    """
    payload = dict(payload)
    payload["retry_count"] = retry_count
    payload["retry_not_before"] = time.time() + delay_s
    await redis.rpush(QUEUE_KEY, json.dumps(payload))
    log.warning(
        "Ingest job re-queued for retry",
        extra={
            "job_id": payload.get("job_id"),
            "retry_count": retry_count,
            "delay_s": delay_s,
        },
    )


async def move_to_dlq(redis: Redis, payload: dict) -> None:
    """Push an exhausted job to the dead-letter queue."""
    await redis.lpush(DLQ_KEY, json.dumps(payload))
    log.error(
        "Ingest job moved to DLQ after max retries",
        extra={"job_id": payload.get("job_id"), "dlq": DLQ_KEY},
    )
