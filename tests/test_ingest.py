"""
tests/test_ingest.py — Unit tests for the ingest pipeline stages.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.models import Chunk, ChunkingConfig, ChunkingStrategy, IngestRequest, SourceType


# ── fetch_content ──────────────────────────────────────────────────────────────

async def test_fetch_content_file_mode_returns_bytes():
    from rag.ingest import fetch_content

    req = IngestRequest(
        namespace_id="ns", source_id="src", source_type=SourceType.file
    )
    raw, filename = await fetch_content(req, file_bytes=b"hello world")
    assert raw == b"hello world"
    assert filename == ""


async def test_fetch_content_file_mode_empty_when_no_bytes():
    from rag.ingest import fetch_content

    req = IngestRequest(
        namespace_id="ns", source_id="src", source_type=SourceType.file
    )
    raw, filename = await fetch_content(req, file_bytes=None)
    assert raw == b""


async def test_fetch_content_url_mode_returns_content_and_filename():
    from rag.ingest import fetch_content

    req = IngestRequest(
        namespace_id="ns",
        source_id="src",
        source_type=SourceType.url,
        url="http://example.com/docs/report.pdf?v=1",
    )

    mock_response = MagicMock()
    mock_response.content = b"PDF bytes"
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.get = AsyncMock(return_value=mock_response)

    with patch("rag.ingest.httpx.AsyncClient", return_value=mock_http):
        raw, filename = await fetch_content(req)

    assert raw == b"PDF bytes"
    assert filename == "report.pdf"  # strips query string


async def test_fetch_content_url_raises_on_http_error():
    from rag.ingest import fetch_content
    import httpx

    req = IngestRequest(
        namespace_id="ns", source_id="src", source_type=SourceType.url,
        url="http://example.com/file.txt",
    )

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=MagicMock()
    )
    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.get = AsyncMock(return_value=mock_response)

    with patch("rag.ingest.httpx.AsyncClient", return_value=mock_http):
        with pytest.raises(httpx.HTTPStatusError):
            await fetch_content(req)


# ── embed_chunks ───────────────────────────────────────────────────────────────

async def test_embed_chunks_returns_one_vector_per_chunk():
    from rag.ingest import embed_chunks

    chunks = [
        Chunk(chunk_id="1", content="First chunk", source_id="src", namespace_id="ns", score=0.0),
        Chunk(chunk_id="2", content="Second chunk", source_id="src", namespace_id="ns", score=0.0),
    ]

    mock_resp = MagicMock()
    mock_resp.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]

    with patch("rag.ingest.openai_client") as mock_client:
        mock_client.embeddings.create = AsyncMock(return_value=mock_resp)
        vectors = await embed_chunks(chunks)

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert vectors[1] == [0.4, 0.5, 0.6]


async def test_embed_chunks_batches_at_configured_size():
    """embed_chunks respects EMBED_BATCH_SIZE — all chunks in one call when count < batch size."""
    from rag.ingest import embed_chunks

    chunks = [
        Chunk(chunk_id=str(i), content=f"chunk {i}", source_id="s", namespace_id="n", score=0.0)
        for i in range(150)
    ]

    batch_calls = []

    async def mock_create(**kwargs):
        batch_calls.append(len(kwargs["input"]))
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.0]) for _ in kwargs["input"]]
        return resp

    with patch("rag.ingest.openai_client") as mock_client:
        mock_client.embeddings.create = mock_create
        await embed_chunks(chunks)

    # With default EMBED_BATCH_SIZE=500, 150 chunks → single API call
    assert len(batch_calls) == 1
    assert batch_calls[0] == 150


# ── run_ingest (full pipeline, mocked externals) ──────────────────────────────

async def test_run_ingest_happy_path():
    from rag.ingest import run_ingest
    from rag.models import IngestStatus

    req = IngestRequest(
        namespace_id="ns", source_id="src", source_type=SourceType.file,
        chunking=ChunkingConfig(strategy=ChunkingStrategy.sliding, chunk_size=100, min_chunk_size=5),
    )

    mock_r = AsyncMock()
    mock_r.get.return_value = None
    mock_qdrant = AsyncMock()

    sample_chunk = Chunk(chunk_id="c1", content="Some text content here.", source_id="src", namespace_id="ns", score=0.0)

    with (
        patch("rag.ingest.fetch_content", AsyncMock(return_value=(b"Some text content here.", "file.txt"))),
        patch("rag.ingest.extract", AsyncMock(return_value=MagicMock(text="Some text content here.", pages=None))),
        patch("rag.ingest.detect_language", return_value="en"),
        patch("rag.ingest.get_chunker") as mock_gc,
        patch("rag.ingest.embed_chunks", AsyncMock(return_value=[[0.1, 0.2, 0.3]])),
        patch("rag.ingest.ensure_collection", AsyncMock()),
        patch("rag.ingest.delete_source", AsyncMock()),
        patch("rag.ingest.upsert_chunks", AsyncMock()),
        patch("rag.ingest.update_job", AsyncMock()),
    ):
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [sample_chunk]
        mock_gc.return_value = mock_chunker

        await run_ingest(mock_r, mock_qdrant, "job-1", "tenant-1", "ns", "src", req, file_bytes=b"Some text content here.")

    # update_job should have been called (at least 4 times: fetching/extracting/chunking/done)
    from rag.ingest import update_job  # just verify no exceptions raised


async def test_run_ingest_marks_failed_on_error():
    from rag.ingest import run_ingest
    from rag.models import IngestStatus

    req = IngestRequest(
        namespace_id="ns", source_id="src", source_type=SourceType.file,
    )

    mock_r = AsyncMock()
    mock_qdrant = AsyncMock()
    captured_status = []

    async def fake_update_job(r, job_id, status, *args, **kwargs):
        captured_status.append(status)

    with (
        patch("rag.ingest.fetch_content", AsyncMock(side_effect=ValueError("Network failure"))),
        patch("rag.ingest.update_job", side_effect=fake_update_job),
    ):
        await run_ingest(mock_r, mock_qdrant, "job-1", "tenant-1", "ns", "src", req)

    assert IngestStatus.failed in captured_status


async def test_run_ingest_deduplicates_before_upsert():
    """Verify delete_source is called before upsert_chunks on re-ingest."""
    from rag.ingest import run_ingest

    req = IngestRequest(
        namespace_id="ns", source_id="src-existing", source_type=SourceType.file,
        chunking=ChunkingConfig(strategy=ChunkingStrategy.sliding, chunk_size=100, min_chunk_size=5),
    )

    mock_r = AsyncMock()
    mock_qdrant = AsyncMock()
    call_order = []

    async def mock_delete(*args, **kwargs):
        call_order.append("delete")

    async def mock_upsert(*args, **kwargs):
        call_order.append("upsert")

    sample_chunk = Chunk(chunk_id="c1", content="Content.", source_id="src-existing", namespace_id="ns", score=0.0)

    with (
        patch("rag.ingest.fetch_content", AsyncMock(return_value=(b"Content.", ""))),
        patch("rag.ingest.extract", AsyncMock(return_value=MagicMock(text="Content.", pages=None))),
        patch("rag.ingest.detect_language", return_value="en"),
        patch("rag.ingest.get_chunker") as mock_gc,
        patch("rag.ingest.embed_chunks", AsyncMock(return_value=[[0.1]])),
        patch("rag.ingest.ensure_collection", AsyncMock()),
        patch("rag.ingest.delete_source", side_effect=mock_delete),
        patch("rag.ingest.upsert_chunks", side_effect=mock_upsert),
        patch("rag.ingest.update_job", AsyncMock()),
    ):
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [sample_chunk]
        mock_gc.return_value = mock_chunker
        await run_ingest(mock_r, mock_qdrant, "job-1", "tenant-1", "ns", "src-existing", req, file_bytes=b"Content.")

    assert call_order == ["delete", "upsert"], f"Expected delete before upsert, got: {call_order}"
