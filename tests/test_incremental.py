"""
tests/test_incremental.py — Tests for incremental ingest mode.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.models import IngestRequest, Chunk, SourceType


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_request(**kwargs) -> IngestRequest:
    defaults = dict(
        namespace_id="ns-test",
        source_id="doc-001",
        source_type=SourceType.file,
    )
    defaults.update(kwargs)
    return IngestRequest(**defaults)


_SAMPLE_CHUNK = Chunk(
    chunk_id="c1", content="Some content.", source_id="doc-001", namespace_id="ns-test", score=0.0
)

_COMMON_PATCHES = {
    "rag.ingest.fetch_content": AsyncMock(return_value=(b"Some content.", "")),
    "rag.ingest.extract": AsyncMock(
        return_value=MagicMock(text="Some content.", pages=None, mime_type="text/plain")
    ),
    "rag.ingest.detect_language": MagicMock(return_value="en"),
    "rag.ingest.embed_chunks": AsyncMock(return_value=[[0.1] * 1536]),
    "rag.ingest.encode_sparse_chunks": MagicMock(return_value=[MagicMock()]),
    "rag.ingest.ensure_collection": AsyncMock(),
    "rag.ingest.upsert_chunks": AsyncMock(),
    "rag.ingest.update_job": AsyncMock(),
}


def _make_chunker_patch():
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [_SAMPLE_CHUNK]
    return mock_chunker


# ── Model field tests ─────────────────────────────────────────────────────────

def test_incremental_field_defaults_to_false():
    req = _make_request()
    assert req.incremental is False


def test_incremental_field_set_to_true():
    req = _make_request(incremental=True)
    assert req.incremental is True


# ── run_ingest behaviour ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_non_incremental_ingest_calls_delete_source():
    """Default (incremental=False) → delete_source called before upsert."""
    from rag.ingest import run_ingest

    req = _make_request(incremental=False)
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_qdrant = AsyncMock()

    with (
        patch("rag.ingest.fetch_content", AsyncMock(return_value=(b"Some content.", ""))),
        patch("rag.ingest.extract", AsyncMock(return_value=MagicMock(text="Some content.", pages=None, mime_type="text/plain"))),
        patch("rag.ingest.detect_language", return_value="en"),
        patch("rag.ingest.get_chunker") as mock_gc,
        patch("rag.ingest.embed_chunks", AsyncMock(return_value=[[0.1] * 1536])),
        patch("rag.ingest.encode_sparse_chunks", return_value=[MagicMock()]),
        patch("rag.ingest.ensure_collection", AsyncMock()),
        patch("rag.ingest.delete_source", AsyncMock()) as mock_delete,
        patch("rag.ingest.upsert_chunks", AsyncMock()),
        patch("rag.ingest.update_job", AsyncMock()),
    ):
        mock_gc.return_value = _make_chunker_patch()
        await run_ingest(mock_redis, mock_qdrant, "job-1", "tenant-x", "ns-test", "doc-001", req)
        mock_delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_incremental_ingest_skips_delete_source():
    """incremental=True → delete_source NOT called."""
    from rag.ingest import run_ingest

    req = _make_request(incremental=True)
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_qdrant = AsyncMock()

    with (
        patch("rag.ingest.fetch_content", AsyncMock(return_value=(b"Some content.", ""))),
        patch("rag.ingest.extract", AsyncMock(return_value=MagicMock(text="Some content.", pages=None, mime_type="text/plain"))),
        patch("rag.ingest.detect_language", return_value="en"),
        patch("rag.ingest.get_chunker") as mock_gc,
        patch("rag.ingest.embed_chunks", AsyncMock(return_value=[[0.1] * 1536])),
        patch("rag.ingest.encode_sparse_chunks", return_value=[MagicMock()]),
        patch("rag.ingest.ensure_collection", AsyncMock()),
        patch("rag.ingest.delete_source", AsyncMock()) as mock_delete,
        patch("rag.ingest.upsert_chunks", AsyncMock()),
        patch("rag.ingest.update_job", AsyncMock()),
    ):
        mock_gc.return_value = _make_chunker_patch()
        await run_ingest(mock_redis, mock_qdrant, "job-1", "tenant-x", "ns-test", "doc-001", req)
        mock_delete.assert_not_awaited()
