"""
tests/test_stream.py — Tests for POST /v1/query/stream (SSE endpoint).
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.models import Chunk, QueryRequest


# ── helpers ────────────────────────────────────────────────────────────────────

def _chunk(content: str) -> Chunk:
    return Chunk(
        chunk_id="c1", content=content,
        source_id="src", namespace_id="ns", score=0.85,
    )


def _parse_events(body: bytes) -> list[dict]:
    """Parse SSE body into list of {event, data} dicts."""
    events = []
    current: dict = {}
    for line in body.decode().splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line[len("data:"):].strip())
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


# ── run_query_stream unit tests ────────────────────────────────────────────────

async def test_stream_no_chunks_emits_citations_then_done():
    """When search returns no chunks, emit citations (empty) + done only."""
    from rag.query import run_query_stream

    req = QueryRequest(question="Q?", namespaces=["ns"])

    events = []
    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[])),
    ):
        async for raw in run_query_stream(AsyncMock(), req, "req-1", "tenant-1"):
            events.append(raw)

    # Collect event names
    event_names = []
    for e in events:
        for line in e.splitlines():
            if line.startswith("event:"):
                event_names.append(line.split(":", 1)[1].strip())

    assert "citations" in event_names
    assert "done" in event_names
    assert "token" not in event_names
    assert "error" not in event_names


async def test_stream_with_chunks_emits_citations_tokens_done():
    """When chunks found, the stream should have citations → tokens → done."""
    from rag.query import run_query_stream

    req = QueryRequest(question="What is X?", namespaces=["ns"], language="en")

    # Minimal streaming mock — yields two deltas then finishes
    async def _fake_stream():
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        chunk1.usage = None
        yield chunk1
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]
        chunk2.usage = None
        yield chunk2

    # StreamingResponse from openai uses async context manager
    fake_stream_cm = MagicMock()
    fake_stream_cm.__aenter__ = AsyncMock(return_value=_fake_stream())
    fake_stream_cm.__aexit__ = AsyncMock(return_value=False)

    collected_events = []
    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[_chunk("Context.")])),
        patch("rag.query.openai_client") as mock_client,
    ):
        mock_client.chat.completions.create = AsyncMock(return_value=fake_stream_cm)
        async for raw in run_query_stream(AsyncMock(), req, "req-2", "tenant-1"):
            collected_events.append(raw)

    event_names = []
    token_values = []
    for e in collected_events:
        for line in e.splitlines():
            if line.startswith("event:"):
                event_names.append(line.split(":", 1)[1].strip())
            if line.startswith("data:") and "token" in event_names[-1:]:
                token_values.append(json.loads(line.split(":", 1)[1].strip()).get("token", ""))

    assert event_names[0] == "citations"
    assert "token" in event_names
    assert event_names[-1] == "done"
    assert "Hello" in token_values
    assert " world" in token_values


async def test_stream_citations_contain_chunk_data():
    """The citations event data must contain chunk content."""
    from rag.query import run_query_stream

    req = QueryRequest(question="Q?", namespaces=["ns"], include_answer=False)

    raw_events = []
    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[_chunk("Important context.")])),
    ):
        # include_answer=False → no LLM call, but we still get citations + done
        async for raw in run_query_stream(AsyncMock(), req, "req-3", "tenant-1"):
            raw_events.append(raw)

    # Find the citations event and parse it
    citations_data = None
    for e in raw_events:
        lines = e.splitlines()
        if any(l == "event: citations" for l in lines):
            for l in lines:
                if l.startswith("data:"):
                    citations_data = json.loads(l[len("data:"):].strip())

    assert citations_data is not None
    assert len(citations_data) == 1
    assert citations_data[0]["chunk"]["content"] == "Important context."


async def test_stream_done_event_has_required_fields():
    """The done event must include request_id, retrieval_strategy, usage, latency_ms."""
    from rag.query import run_query_stream

    req = QueryRequest(question="Q?", namespaces=["ns"])

    raw_events = []
    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[])),
    ):
        async for raw in run_query_stream(AsyncMock(), req, "req-done", "tenant-1"):
            raw_events.append(raw)

    done_data = None
    for e in raw_events:
        lines = e.splitlines()
        if any(l == "event: done" for l in lines):
            for l in lines:
                if l.startswith("data:"):
                    done_data = json.loads(l[len("data:"):].strip())

    assert done_data is not None
    assert done_data["request_id"] == "req-done"
    assert "retrieval_strategy" in done_data
    assert "usage" in done_data
    assert "latency_ms" in done_data


async def test_stream_emits_error_event_on_prepare_failure():
    """If embed/search fails, the stream should emit a single error event."""
    from rag.query import run_query_stream

    req = QueryRequest(question="Q?", namespaces=["ns"])

    raw_events = []
    with patch("rag.query.embed_question", AsyncMock(side_effect=RuntimeError("embed failed"))):
        async for raw in run_query_stream(AsyncMock(), req, "req-err", "tenant-1"):
            raw_events.append(raw)

    event_names = []
    for e in raw_events:
        for line in e.splitlines():
            if line.startswith("event:"):
                event_names.append(line.split(":", 1)[1].strip())

    assert event_names == ["error"]


# ── HTTP endpoint tests (via test client) ─────────────────────────────────────

async def test_stream_endpoint_requires_auth(client):
    resp = await client.post(
        "/v1/query/stream",
        json={"question": "Hello?", "namespaces": ["ns"]},
    )
    assert resp.status_code == 401


async def test_stream_endpoint_returns_event_stream(client):
    req_body = {"question": "What is X?", "namespaces": ["ns"]}

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[])),
    ):
        resp = await client.post(
            "/v1/query/stream",
            json=req_body,
            headers={
                "Authorization": "Bearer test-api-key-12345",
                "X-Request-ID": "req-stream-001",
                "X-Tenant-ID":  "tenant-test",
            },
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


async def test_stream_endpoint_body_contains_done_event(client):
    req_body = {"question": "What is X?", "namespaces": ["ns"]}

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[])),
    ):
        resp = await client.post(
            "/v1/query/stream",
            json=req_body,
            headers={
                "Authorization": "Bearer test-api-key-12345",
                "X-Request-ID": "req-stream-002",
                "X-Tenant-ID":  "tenant-test",
            },
        )

    events = _parse_events(resp.content)
    event_names = [e.get("event") for e in events]
    assert "citations" in event_names
    assert "done" in event_names
