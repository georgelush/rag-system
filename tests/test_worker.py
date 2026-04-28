"""
tests/test_worker.py — Unit tests for rag/worker.py retry & DLQ logic.

Covers:
  - _is_retryable: returns False for non-retryable exception types / messages
  - _is_retryable: returns True for generic runtime errors
  - _process_one: retryable error → enqueue_retry called + INGEST_RETRIES incremented
  - _process_one: non-retryable error (ValueError) → move_to_dlq, no retry
  - _process_one: max retries exceeded → move_to_dlq
  - _process_one: successful job → no retry, no DLQ
  - _process_one: no job in queue → returns False
  - _process_one: job with future retry_not_before → re-queued without processing
"""

import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── _is_retryable unit tests ──────────────────────────────────────────────────

def test_is_retryable_true_for_runtime_error():
    from rag.worker import _is_retryable
    assert _is_retryable(RuntimeError("network timeout")) is True


def test_is_retryable_true_for_generic_exception():
    from rag.worker import _is_retryable
    assert _is_retryable(Exception("some transient failure")) is True


def test_is_retryable_false_for_value_error():
    from rag.worker import _is_retryable
    assert _is_retryable(ValueError("bad input")) is False


def test_is_retryable_false_for_not_implemented_error():
    from rag.worker import _is_retryable
    assert _is_retryable(NotImplementedError("not supported")) is False


def test_is_retryable_false_for_type_error():
    from rag.worker import _is_retryable
    assert _is_retryable(TypeError("wrong type")) is False


def test_is_retryable_false_for_empty_content_message():
    from rag.worker import _is_retryable
    assert _is_retryable(RuntimeError("empty content after extraction")) is False


def test_is_retryable_false_for_unsupported_mime_message():
    from rag.worker import _is_retryable
    assert _is_retryable(RuntimeError("unsupported mime type: application/octet-stream")) is False


def test_is_retryable_false_for_unsupported_format_message():
    from rag.worker import _is_retryable
    assert _is_retryable(RuntimeError("unsupported format")) is False


def test_is_retryable_false_for_validation_message():
    from rag.worker import _is_retryable
    assert _is_retryable(RuntimeError("validation error in payload")) is False


def test_is_retryable_message_check_is_case_insensitive():
    from rag.worker import _is_retryable
    # uppercase / mixed case should still be caught
    assert _is_retryable(RuntimeError("EMPTY CONTENT detected")) is False


# ── _process_one integration-style tests ────────────────────────────────────

def _sample_job(retry_count: int = 0, retry_not_before: float = 0.0) -> dict:
    return {
        "job_id": "job-abc",
        "tenant_id": "tenant-1",
        "namespace_id": "ns-1",
        "source_id": "src-1",
        "retry_count": retry_count,
        "retry_not_before": retry_not_before,
        "file_bytes_hex": None,
        "req": {
            "source_type": "file",
            "source_id": "src-1",
            "namespace_id": "ns-1",
        },
    }


def _make_redis_with_job(job: dict | None) -> AsyncMock:
    r = AsyncMock()
    raw = json.dumps(job).encode() if job else None
    # dequeue_ingest calls brpop → returns (key, value) tuple
    if raw is not None:
        r.brpop = AsyncMock(return_value=(b"rag:queue:ingest", raw))
    else:
        r.brpop = AsyncMock(return_value=None)
    r.rpush = AsyncMock(return_value=1)
    r.lpush = AsyncMock(return_value=1)
    r.get = AsyncMock(return_value=None)
    r.set = AsyncMock(return_value=True)
    r.setex = AsyncMock(return_value=True)
    return r


@pytest.mark.asyncio
async def test_process_one_returns_false_when_queue_empty():
    from rag.worker import _process_one
    redis = AsyncMock()
    redis.brpop = AsyncMock(return_value=None)
    qdrant = AsyncMock()
    result = await _process_one(redis, qdrant)
    assert result is False


@pytest.mark.asyncio
async def test_process_one_retryable_error_calls_enqueue_retry():
    """RuntimeError (retryable) on first attempt → enqueue_retry is called."""
    from rag.worker import _process_one
    from rag.metrics import INGEST_RETRIES

    job = _sample_job(retry_count=0)
    redis = _make_redis_with_job(job)
    qdrant = AsyncMock()

    with patch("rag.worker.run_ingest", new_callable=AsyncMock) as mock_run, \
         patch("rag.worker.enqueue_retry", new_callable=AsyncMock) as mock_retry, \
         patch("rag.worker.move_to_dlq", new_callable=AsyncMock) as mock_dlq:

        mock_run.side_effect = RuntimeError("transient network error")
        before = INGEST_RETRIES._value.get()
        result = await _process_one(redis, qdrant)

    assert result is True
    mock_retry.assert_awaited_once()
    mock_dlq.assert_not_awaited()
    assert INGEST_RETRIES._value.get() == before + 1


@pytest.mark.asyncio
async def test_process_one_non_retryable_goes_to_dlq_immediately():
    """ValueError (non-retryable) → move_to_dlq directly, no retry."""
    from rag.worker import _process_one

    job = _sample_job(retry_count=0)
    redis = _make_redis_with_job(job)
    qdrant = AsyncMock()

    with patch("rag.worker.run_ingest", new_callable=AsyncMock) as mock_run, \
         patch("rag.worker.enqueue_retry", new_callable=AsyncMock) as mock_retry, \
         patch("rag.worker.move_to_dlq", new_callable=AsyncMock) as mock_dlq:

        mock_run.side_effect = ValueError("bad content")
        await _process_one(redis, qdrant)

    mock_dlq.assert_awaited_once()
    mock_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_one_max_retries_exceeded_goes_to_dlq():
    """retry_count=MAX_RETRIES → next_retry > MAX_RETRIES → DLQ."""
    from rag.worker import _process_one, MAX_RETRIES

    job = _sample_job(retry_count=MAX_RETRIES)  # already at max
    redis = _make_redis_with_job(job)
    qdrant = AsyncMock()

    with patch("rag.worker.run_ingest", new_callable=AsyncMock) as mock_run, \
         patch("rag.worker.enqueue_retry", new_callable=AsyncMock) as mock_retry, \
         patch("rag.worker.move_to_dlq", new_callable=AsyncMock) as mock_dlq:

        mock_run.side_effect = RuntimeError("still failing")
        await _process_one(redis, qdrant)

    mock_dlq.assert_awaited_once()
    mock_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_one_success_no_retry_no_dlq():
    """Successful run_ingest → neither enqueue_retry nor move_to_dlq called."""
    from rag.worker import _process_one

    job = _sample_job(retry_count=0)
    redis = _make_redis_with_job(job)
    qdrant = AsyncMock()

    with patch("rag.worker.run_ingest", new_callable=AsyncMock) as mock_run, \
         patch("rag.worker.enqueue_retry", new_callable=AsyncMock) as mock_retry, \
         patch("rag.worker.move_to_dlq", new_callable=AsyncMock) as mock_dlq:

        mock_run.return_value = None
        result = await _process_one(redis, qdrant)

    assert result is True
    mock_retry.assert_not_awaited()
    mock_dlq.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_one_respects_retry_not_before():
    """Job with future retry_not_before is re-queued without processing."""
    from rag.worker import _process_one

    future_time = time.time() + 3600  # 1 hour in future
    job = _sample_job(retry_count=1, retry_not_before=future_time)
    redis = _make_redis_with_job(job)
    qdrant = AsyncMock()

    with patch("rag.worker.run_ingest", new_callable=AsyncMock) as mock_run:
        result = await _process_one(redis, qdrant)

    mock_run.assert_not_awaited()
    # Job re-queued (rpush called)
    redis.rpush.assert_awaited_once()
    assert result is False


@pytest.mark.asyncio
async def test_process_one_empty_content_message_goes_to_dlq():
    """'empty content' message in RuntimeError → non-retryable → DLQ."""
    from rag.worker import _process_one

    job = _sample_job(retry_count=0)
    redis = _make_redis_with_job(job)
    qdrant = AsyncMock()

    with patch("rag.worker.run_ingest", new_callable=AsyncMock) as mock_run, \
         patch("rag.worker.enqueue_retry", new_callable=AsyncMock) as mock_retry, \
         patch("rag.worker.move_to_dlq", new_callable=AsyncMock) as mock_dlq:

        mock_run.side_effect = RuntimeError("empty content after strip")
        await _process_one(redis, qdrant)

    mock_dlq.assert_awaited_once()
    mock_retry.assert_not_awaited()
