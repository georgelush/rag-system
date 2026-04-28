"""
src/rag/query.py — Query pipeline for POST /v1/query and POST /v1/query/stream.
"""

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from redis.asyncio import Redis
from qdrant_client import AsyncQdrantClient
import httpx

from rag.models import (
    Citation,
    ConversationTurn,
    QueryRequest,
    QueryResponse,
    Usage,
)

from rag.vector_store import collection_name, search_chunks, EMBEDDING_MODEL
from rag.ingest import embed_chunks
from rag.language import detect_language, language_name
from rag.openai_client import openai_client
from rag.sparse_encoder import encode as encode_sparse
from rag.cost import llm_cost_usd, embedding_cost_usd
from rag.cache import get_cached, set_cached
from rag import telemetry
from rag.metrics import RERANKER_FAILURES, NAMESPACE_QUERIES
from rag.config import RERANKER_TIMEOUT_S

log = logging.getLogger(__name__)

LLM_MODEL    = os.environ.get("LLM_MODEL",    "gpt-4o-mini")
RERANKER_URL = os.environ.get("RERANKER_URL", "")


async def _rerank_chunks(chunks: list, query: str) -> list:
    """Call the reranker sidecar to reorder *chunks* by cross-encoder score.

    Falls back to the original score-sorted order on any error (network,
    timeout, etc.) so a reranker outage never breaks query responses.
    Returns the original list unchanged when RERANKER_URL is not configured.
    """
    if not RERANKER_URL or not chunks:
        return chunks
    try:
        async with httpx.AsyncClient(timeout=RERANKER_TIMEOUT_S) as client:
            resp = await client.post(
                f"{RERANKER_URL.rstrip('/')}/rerank",
                json={
                    "query":    query,
                    "passages": [
                        {"id": c.chunk_id, "text": c.content} for c in chunks
                    ],
                },
            )
            resp.raise_for_status()
    except Exception as exc:
        RERANKER_FAILURES.inc()
        log.warning(
            "Reranker call failed — using original ranking",
            extra={"error": str(exc)},
        )
        return chunks

    score_map = {p["id"]: p["score"] for p in resp.json()["passages"]}
    return sorted(
        chunks,
        key=lambda c: score_map.get(c.chunk_id, c.score),
        reverse=True,
    )


async def embed_question(question: str) -> list[float]:
    """Embed the user question using the same model as ingest."""
    response = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    )
    return response.data[0].embedding


def build_prompt(
    question: str,
    chunks: list,
    history: list[ConversationTurn],
    style_hints,
    language: str = "en",
) -> list[dict]:
    """Build OpenAI messages list for answer generation."""
    lang_name   = language_name(language)
    max_chars   = style_hints.answer_max_chars if style_hints else 2000
    cite_inline = style_hints.cite_inline if style_hints else True
    tone        = style_hints.tone.value if style_hints and style_hints.tone else "formal"

    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(f"[{i}]: {chunk.content}")

    context_text = "\n\n".join(context_blocks)

    cite_instruction = (
        "Insert citation markers [1], [2] etc. directly in the text after each claim."
        if cite_inline else
        "Do not insert citation markers in the text."
    )

    system = (
        f"You are a helpful assistant. Answer ONLY based on the provided context. "
        f"Respond in {lang_name}. Tone: {tone}. Maximum {max_chars} characters. "
        f"Plain text only — no Markdown, no asterisks, no formatting. "
        f"{cite_instruction}"
    )

    messages = [{"role": "system", "content": system}]

    for turn in history[-15:]:
        messages.append({"role": turn.role, "content": turn.content})

    messages.append({
        "role": "user",
        "content": f"Question: {question}\n\nContext:\n{context_text}",
    })

    return messages


async def run_query(
    qdrant: AsyncQdrantClient,
    req: QueryRequest,
    request_id: str,
    tenant_id: str,
    redis: Redis | None = None,
) -> QueryResponse:
    """Execute the full query pipeline. Returns QueryResponse; empty result when no chunks found."""
    t_start = time.monotonic()

    # Cache check — skip expensive pipeline on hit
    if redis is not None and req.include_answer:
        cached = await get_cached(redis, tenant_id, req.question, req.namespaces, req.top_k)
        if cached is not None:
            cached["request_id"] = request_id  # freshen request_id on hit
            return QueryResponse.model_validate(cached)

    # Start Langfuse trace (no-op if keys not configured)
    trace = telemetry.start_trace(
        "query",
        trace_id=request_id,
        user_id=tenant_id,
        metadata={
            "namespaces": req.namespaces,
            "top_k": req.top_k,
            "model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    # Step 1: resolve language — caller hint or auto-detect from question
    language: str = req.language or detect_language(req.question)

    # Step 2: embed question + encode sparse vector for hybrid search
    span_embed = trace.span(name="embed_question", input={"question": req.question})
    query_vector = await embed_question(req.question)
    sparse_vector = encode_sparse(req.question)
    embed_tokens = len(req.question.split())  # rough estimate — not billed per token by OpenAI embeddings
    embed_cost = embedding_cost_usd(EMBEDDING_MODEL, embed_tokens)
    span_embed.end(output={"vector_dims": len(query_vector), "sparse_nnz": len(sparse_vector.indices)})

    # Step 3: search all requested namespaces (hybrid or dense-only per collection)
    span_search = trace.span(
        name="search_chunks",
        input={"namespaces": req.namespaces, "top_k": req.top_k},
    )
    all_chunks = []
    for ns_id in req.namespaces:
        NAMESPACE_QUERIES.labels(namespace_id=ns_id).inc()
        col = collection_name(tenant_id, ns_id)
        chunks = await search_chunks(
            qdrant, col, query_vector,
            top_k=req.top_k,
            hint_page_number=req.hint_page_number,
            sparse_vector=sparse_vector,
        )
        all_chunks.extend(chunks)

    # Sort by score descending, keep top_k globally
    all_chunks.sort(key=lambda c: c.score, reverse=True)
    all_chunks = all_chunks[: req.top_k]

    retrieval_strategy = (
        "hybrid+page_boost" if req.hint_page_number else "hybrid"
    ) if sparse_vector.indices else (
        "semantic+page_boost" if req.hint_page_number else "semantic"
    )

    # Optional cross-encoder reranking via sidecar
    if req.rerank and RERANKER_URL and all_chunks:
        all_chunks = await _rerank_chunks(all_chunks, req.question)
        retrieval_strategy = retrieval_strategy + "+reranked"

    span_search.end(output={"chunks_found": len(all_chunks), "strategy": retrieval_strategy})

    # Step 4: empty result guard — no hallucination
    if not all_chunks:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        log.info(
            "Query returned no results",
            extra={"request_id": request_id, "tenant_id": tenant_id, "latency_ms": latency_ms},
        )
        return QueryResponse(
            request_id=request_id,
            answer=None,
            citations=[],
            usage=Usage(
                input_tokens=0, output_tokens=0,
                cost_usd=round(embed_cost, 8), model_id=LLM_MODEL,
            ),
            latency_ms=latency_ms,
            model_version=LLM_MODEL,
            retrieval_strategy=retrieval_strategy,
            confidence=0.0,
            trace_id=request_id,
        )

    # Step 5: build citations
    citations = [
        Citation(marker=f"[{i}]", chunk=chunk)
        for i, chunk in enumerate(all_chunks, 1)
    ]

    # Step 6: generate answer (skip if include_answer=false)
    answer: str | None = None
    input_tokens = 0
    output_tokens = 0
    llm_cost = 0.0

    if req.include_answer:
        messages = build_prompt(
            req.question, all_chunks,
            req.conversation_history,
            req.style_hints,
            language=language,
        )
        span_gen = trace.generation(
            name="generate_answer",
            model=LLM_MODEL,
            input=messages,
        )
        completion = await openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
        )
        answer = completion.choices[0].message.content
        input_tokens  = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        llm_cost      = llm_cost_usd(LLM_MODEL, input_tokens, output_tokens)

        span_gen.end(
            output=answer,
            usage={
                "input":  input_tokens,
                "output": output_tokens,
            },
        )

    total_cost = round(embed_cost + llm_cost, 8)
    latency_ms = int((time.monotonic() - t_start) * 1000)

    log.info(
        "Query completed",
        extra={
            "request_id":   request_id,
            "tenant_id":    tenant_id,
            "latency_ms":   latency_ms,
            "chunks_found": len(all_chunks),
            "strategy":     retrieval_strategy,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":     total_cost,
        },
    )

    response = QueryResponse(
        request_id=request_id,
        answer=answer,
        citations=citations,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=total_cost,
            model_id=LLM_MODEL,
        ),
        latency_ms=latency_ms,
        model_version=LLM_MODEL,
        retrieval_strategy=retrieval_strategy,
        confidence=round(all_chunks[0].score, 2) if all_chunks else 0.0,
        trace_id=request_id,
    )

    # Cache the result for subsequent identical queries
    if redis is not None and req.include_answer and answer is not None:
        await set_cached(redis, tenant_id, req.question, req.namespaces, req.top_k, response.model_dump())

    return response


# ── Streaming pipeline ─────────────────────────────────────────────────────────

async def _prepare_stream(
    qdrant: AsyncQdrantClient,
    req: QueryRequest,
    request_id: str,
    tenant_id: str,
) -> tuple[list[Citation], list[dict], str, str, float]:
    """Shared setup for streaming: embed → search → build_prompt.

    Returns:
        (citations, messages, retrieval_strategy, language, embed_cost)
    """
    language: str = req.language or detect_language(req.question)

    query_vector  = await embed_question(req.question)
    sparse_vector = encode_sparse(req.question)
    embed_tokens  = len(req.question.split())
    embed_cost    = embedding_cost_usd(EMBEDDING_MODEL, embed_tokens)

    all_chunks = []
    for ns_id in req.namespaces:
        col = collection_name(tenant_id, ns_id)
        chunks = await search_chunks(
            qdrant, col, query_vector,
            top_k=req.top_k,
            hint_page_number=req.hint_page_number,
            sparse_vector=sparse_vector,
        )
        all_chunks.extend(chunks)

    all_chunks.sort(key=lambda c: c.score, reverse=True)
    all_chunks = all_chunks[: req.top_k]

    retrieval_strategy = (
        "hybrid+page_boost" if req.hint_page_number else "hybrid"
    ) if sparse_vector.indices else (
        "semantic+page_boost" if req.hint_page_number else "semantic"
    )

    # Optional cross-encoder reranking via sidecar
    if req.rerank and RERANKER_URL and all_chunks:
        all_chunks = await _rerank_chunks(all_chunks, req.question)
        retrieval_strategy = retrieval_strategy + "+reranked"

    citations = [
        Citation(marker=f"[{i}]", chunk=chunk)
        for i, chunk in enumerate(all_chunks, 1)
    ]

    messages = build_prompt(
        req.question, all_chunks,
        req.conversation_history,
        req.style_hints,
        language=language,
    ) if all_chunks else []

    return citations, messages, retrieval_strategy, language, embed_cost


async def run_query_stream(
    qdrant: AsyncQdrantClient,
    req: QueryRequest,
    request_id: str,
    tenant_id: str,
) -> AsyncIterator[str]:
    """Stream query response as Server-Sent Events.

    Event sequence:
      1. ``citations`` — JSON array of Citation objects (sent immediately after retrieval)
      2. ``token``     — one SSE per streamed token from the LLM
      3. ``done``      — final metadata: usage, latency_ms, retrieval_strategy, cost_usd
      4. ``error``     — sent instead of ``done`` if an exception occurs

    Each SSE line format::

        data: <json>\\n\\n

    Callers detect end-of-stream by the ``done`` or ``error`` event type.
    """
    t_start = time.monotonic()

    try:
        citations, messages, retrieval_strategy, language, embed_cost = await _prepare_stream(
            qdrant, req, request_id, tenant_id
        )
    except Exception as exc:
        log.error("Stream prepare failed", extra={"request_id": request_id, "error": str(exc)}, exc_info=True)
        yield f"event: error\ndata: {json.dumps({'code': 'internal_error', 'message': str(exc)})}\n\n"
        return

    # Event 1: send citations immediately (before LLM starts)
    yield (
        f"event: citations\n"
        f"data: {json.dumps([c.model_dump() for c in citations])}\n\n"
    )

    if not messages:
        # No chunks found — send empty done event
        latency_ms = int((time.monotonic() - t_start) * 1000)
        yield (
            f"event: done\n"
            f"data: {json.dumps({'request_id': request_id, 'retrieval_strategy': retrieval_strategy, 'usage': {'input_tokens': 0, 'output_tokens': 0, 'cost_usd': round(embed_cost, 8), 'model_id': LLM_MODEL}, 'latency_ms': latency_ms})}\n\n"
        )
        return

    # Event 2+: stream LLM tokens
    input_tokens  = 0
    output_tokens = 0
    answer_parts: list[str] = []

    try:
        async with await openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            stream=True,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    token = delta.content
                    answer_parts.append(token)
                    output_tokens += 1  # approximate — OpenAI doesn't send per-token usage in stream
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                # Capture usage from final chunk if provider sends it
                if chunk.usage:
                    input_tokens  = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
    except Exception as exc:
        log.error("Stream LLM failed", extra={"request_id": request_id, "error": str(exc)}, exc_info=True)
        yield f"event: error\ndata: {json.dumps({'code': 'upstream_error', 'message': str(exc)})}\n\n"
        return

    # Estimate input tokens if provider didn't supply usage in the stream
    if input_tokens == 0 and messages:
        input_tokens = sum(len(m.get("content", "").split()) for m in messages)

    llm_cost   = llm_cost_usd(LLM_MODEL, input_tokens, output_tokens)
    total_cost = round(embed_cost + llm_cost, 8)
    latency_ms = int((time.monotonic() - t_start) * 1000)

    log.info(
        "Stream query completed",
        extra={
            "request_id":    request_id,
            "tenant_id":     tenant_id,
            "latency_ms":    latency_ms,
            "output_tokens": output_tokens,
            "cost_usd":      total_cost,
        },
    )

    # Event 3: done
    yield (
        f"event: done\n"
        f"data: {json.dumps({'request_id': request_id, 'retrieval_strategy': retrieval_strategy, 'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'cost_usd': total_cost, 'model_id': LLM_MODEL}, 'latency_ms': latency_ms})}\n\n"
    )

