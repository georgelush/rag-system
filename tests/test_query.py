"""
tests/test_query.py — Unit tests for the query pipeline.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.models import (
    Chunk,
    ConversationTurn,
    QueryRequest,
    StyleHints,
    Tone,
)
from rag.query import build_prompt


# ── build_prompt ───────────────────────────────────────────────────────────────

def _chunk(content: str) -> Chunk:
    return Chunk(chunk_id="1", content=content, source_id="src", namespace_id="ns", score=0.9)


def test_build_prompt_system_is_english_by_default():
    msgs = build_prompt("What is X?", [_chunk("Context.")], [], None, language="en")
    system = msgs[0]["content"]
    assert "English" in system
    assert "Ești" not in system
    assert "juridic" not in system


def test_build_prompt_system_uses_detected_language():
    msgs = build_prompt("Was ist X?", [_chunk("Kontext.")], [], None, language="de")
    system = msgs[0]["content"]
    assert "German" in system


def test_build_prompt_system_uses_french():
    msgs = build_prompt("Qu'est-ce?", [_chunk("Contenu.")], [], None, language="fr")
    assert "French" in msgs[0]["content"]


def test_build_prompt_user_message_contains_question():
    msgs = build_prompt("My question here", [_chunk("Context.")], [], None, language="en")
    user_msg = msgs[-1]["content"]
    assert "My question here" in user_msg


def test_build_prompt_context_uses_numeric_markers():
    chunks = [_chunk("First content"), _chunk("Second content")]
    msgs = build_prompt("Q?", chunks, [], None, language="en")
    user_msg = msgs[-1]["content"]
    assert "[1]:" in user_msg
    assert "[2]:" in user_msg
    assert "First content" in user_msg
    assert "Second content" in user_msg


def test_build_prompt_no_romanian_labels():
    msgs = build_prompt("Test?", [_chunk("Content.")], [], None, language="en")
    full = " ".join(m["content"] for m in msgs)
    assert "Întrebare" not in full
    assert "Texte juridice" not in full
    assert "Art." not in full


def test_build_prompt_includes_conversation_history():
    history = [
        ConversationTurn(role="user", content="Previous question"),
        ConversationTurn(role="assistant", content="Previous answer"),
    ]
    msgs = build_prompt("Follow-up?", [_chunk("Context.")], history, None, language="en")
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    assert roles.count("user") >= 2
    assert "assistant" in roles


def test_build_prompt_limits_history_to_15():
    history = [ConversationTurn(role="user", content=f"Q{i}") for i in range(20)]
    msgs = build_prompt("Final?", [_chunk("Context.")], history, None, language="en")
    user_turns = [m for m in msgs if m["role"] == "user"]
    # 15 history user turns + 1 current question = 16 max
    assert len(user_turns) <= 16


def test_build_prompt_cite_inline_instruction():
    hints = StyleHints(cite_inline=True)
    msgs = build_prompt("Q?", [_chunk("Content.")], [], hints, language="en")
    system = msgs[0]["content"]
    assert "[1]" in system or "citation" in system.lower()


def test_build_prompt_no_cite_inline():
    hints = StyleHints(cite_inline=False)
    msgs = build_prompt("Q?", [_chunk("Content.")], [], hints, language="en")
    system = msgs[0]["content"]
    assert "Do not insert" in system


def test_build_prompt_tone_in_system():
    hints = StyleHints(tone=Tone.casual)
    msgs = build_prompt("Q?", [_chunk("Content.")], [], hints, language="en")
    assert "casual" in msgs[0]["content"]


def test_build_prompt_max_chars_in_system():
    hints = StyleHints(answer_max_chars=500)
    msgs = build_prompt("Q?", [_chunk("Content.")], [], hints, language="en")
    assert "500" in msgs[0]["content"]


# ── embed_question ─────────────────────────────────────────────────────────────

async def test_embed_question_returns_vector():
    from rag.query import embed_question

    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

    with patch("rag.query.openai_client") as mock_client:
        mock_client.embeddings.create = AsyncMock(return_value=mock_resp)
        vector = await embed_question("What is the meaning of life?")

    assert vector == [0.1, 0.2, 0.3]


# ── run_query ──────────────────────────────────────────────────────────────────

async def test_run_query_returns_empty_when_no_chunks():
    from rag.query import run_query

    req = QueryRequest(
        question="What is X?",
        namespaces=["ns"],
        include_answer=False,
    )

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[])),
    ):
        result = await run_query(AsyncMock(), req, "req-001", "tenant-1")

    assert result.answer is None
    assert result.citations == []
    assert result.confidence == 0.0


async def test_run_query_auto_detects_language():
    from rag.query import run_query

    req = QueryRequest(
        question="Was bedeutet das?",  # German
        namespaces=["ns"],
        include_answer=False,
        language=None,  # auto-detect
    )

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[])),
    ):
        result = await run_query(AsyncMock(), req, "req-001", "tenant-1")

    # Should not crash — language detection ran without error
    assert result is not None


async def test_run_query_returns_citations_for_found_chunks():
    from rag.query import run_query

    chunk = Chunk(
        chunk_id="c1", content="The answer is 42.",
        source_id="src", namespace_id="ns", score=0.95,
    )

    req = QueryRequest(
        question="What is the answer?",
        namespaces=["ns"],
        include_answer=False,  # skip LLM call
    )

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[chunk])),
    ):
        result = await run_query(AsyncMock(), req, "req-001", "tenant-1")

    assert len(result.citations) == 1
    assert result.citations[0].chunk.content == "The answer is 42."
    assert result.confidence == 0.95


async def test_run_query_uses_hint_page_number():
    from rag.query import run_query

    req = QueryRequest(
        question="What is on page 5?",
        namespaces=["ns"],
        hint_page_number=5,
        include_answer=False,
    )

    captured_kwargs = {}

    async def mock_search(qdrant, col, vector, top_k, hint_page_number=None, sparse_vector=None):
        captured_kwargs["hint_page_number"] = hint_page_number
        return []

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", side_effect=mock_search),
    ):
        await run_query(AsyncMock(), req, "req-001", "tenant-1")

    assert captured_kwargs["hint_page_number"] == 5


async def test_run_query_generates_answer_when_include_answer_true():
    from rag.query import run_query

    chunk = Chunk(
        chunk_id="c1", content="The capital is Paris.",
        source_id="src", namespace_id="ns", score=0.9,
    )

    req = QueryRequest(
        question="What is the capital?",
        namespaces=["ns"],
        include_answer=True,
        language="en",
    )

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="The capital is Paris."))]
    mock_completion.usage = MagicMock(prompt_tokens=50, completion_tokens=10)

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[chunk])),
        patch("rag.query.openai_client") as mock_client,
    ):
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        result = await run_query(AsyncMock(), req, "req-001", "tenant-1")

    assert result.answer == "The capital is Paris."
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 10


async def test_run_query_retrieval_strategy_label():
    from rag.query import run_query

    req_plain = QueryRequest(question="Q?", namespaces=["ns"], include_answer=False)
    req_paged = QueryRequest(question="Q?", namespaces=["ns"], hint_page_number=3, include_answer=False)

    with (
        patch("rag.query.embed_question", AsyncMock(return_value=[0.0] * 1536)),
        patch("rag.query.search_chunks", AsyncMock(return_value=[])),
    ):
        r1 = await run_query(AsyncMock(), req_plain, "req-1", "t")
        r2 = await run_query(AsyncMock(), req_paged, "req-2", "t")

    assert r1.retrieval_strategy == "semantic"
    assert r2.retrieval_strategy == "semantic+page_boost"
