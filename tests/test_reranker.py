"""
tests/test_reranker.py — Tests for reranker integration in the query pipeline.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.models import Chunk, QueryRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(chunk_id: str, content: str, score: float = 0.5) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, content=content,
        source_id="src", namespace_id="ns", score=score,
    )


# ── _rerank_chunks unit tests ─────────────────────────────────────────────────

async def test_rerank_returns_original_when_url_not_set():
    """No RERANKER_URL → return chunks unchanged."""
    from rag.query import _rerank_chunks
    chunks = [_chunk("a", "text a"), _chunk("b", "text b")]
    with patch("rag.query.RERANKER_URL", ""):
        result = await _rerank_chunks(chunks, "query?")
    assert result == chunks


async def test_rerank_returns_original_when_empty():
    """Empty chunk list → return immediately, no HTTP call."""
    from rag.query import _rerank_chunks
    with patch("rag.query.RERANKER_URL", "http://reranker:8090"):
        result = await _rerank_chunks([], "query?")
    assert result == []


async def test_rerank_reorders_chunks_by_sidecar_scores():
    """Reranker response with reversed scores should reorder chunks."""
    import httpx
    from rag.query import _rerank_chunks

    chunks = [_chunk("a", "low relevance", score=0.9), _chunk("b", "high relevance", score=0.5)]

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "passages": [
            {"id": "b", "score": 0.95},   # b should come first after reranking
            {"id": "a", "score": 0.20},
        ]
    }

    with (
        patch("rag.query.RERANKER_URL", "http://reranker:8090"),
        patch("httpx.AsyncClient") as MockClient,
    ):
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)
        mock_http.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        result = await _rerank_chunks(chunks, "high relevance")

    assert result[0].chunk_id == "b"
    assert result[1].chunk_id == "a"


async def test_rerank_falls_back_on_http_error():
    """Network error → return original order, no exception raised."""
    import httpx
    from rag.query import _rerank_chunks

    chunks = [_chunk("a", "text a"), _chunk("b", "text b")]

    with (
        patch("rag.query.RERANKER_URL", "http://reranker:8090"),
        patch("httpx.AsyncClient") as MockClient,
    ):
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)
        mock_http.post = AsyncMock(side_effect=Exception("connection refused"))
        MockClient.return_value = mock_http

        result = await _rerank_chunks(chunks, "query?")

    assert result == chunks  # original order preserved


# ── Integration: retrieval_strategy updated when reranker is active ───────────

async def test_retrieval_strategy_includes_reranked_suffix():
    """run_query should append '+reranked' to strategy when reranker is active."""
    from rag.query import run_query

    chunk = _chunk("c1", "Relevant content", score=0.8)
    req = QueryRequest(question="Q?", namespaces=["ns"], include_answer=False)

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"passages": [{"id": "c1", "score": 0.9}]}

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[chunk])),
        patch("rag.query.RERANKER_URL", "http://reranker:8090"),
        patch("httpx.AsyncClient") as MockClient,
    ):
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)
        mock_http.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        result = await run_query(AsyncMock(), req, "req-1", "tenant-1")

    assert "+reranked" in result.retrieval_strategy


async def test_rerank_false_skips_reranker():
    """rerank=false should skip the reranker even when RERANKER_URL is set."""
    from rag.query import run_query

    chunk = _chunk("c1", "Content", score=0.8)
    req = QueryRequest(question="Q?", namespaces=["ns"], include_answer=False, rerank=False)

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[chunk])),
        patch("rag.query.RERANKER_URL", "http://reranker:8090"),
        patch("httpx.AsyncClient") as MockClient,
    ):
        result = await run_query(AsyncMock(), req, "req-1", "tenant-1")
        # httpx should NOT have been called
        MockClient.assert_not_called()

    assert "+reranked" not in (result.retrieval_strategy or "")
