"""
src/rag/models.py — Pydantic schemas compliant with rag-api-spec.yaml v1.0

Defines all request/response shapes used by the RAG service:
  - Chunk, Citation, Usage                  → core data structures
  - QueryRequest / QueryResponse            → POST /v1/query
  - IngestRequest / IngestJob               → POST /v1/ingest
  - NamespaceStats                          → GET /v1/namespaces/{id}/stats
  - HealthStatus                            → GET /v1/health
  - ErrorDetail / ErrorResponse             → any non-2xx response
  - DeleteJobResponse                       → DELETE /v1/namespaces/{id}
"""

from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator

# Shared identifier constraints (namespace_id, source_id in input payloads)
_NS_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]{0,255}$"

# ── Enums ──────────────────────────────────────────────────────────────────────

class IngestStatus(str, Enum):
    queued     = "queued"
    fetching   = "fetching"
    extracting = "extracting"
    chunking   = "chunking"
    embedding  = "embedding"
    indexing   = "indexing"
    done       = "done"
    failed     = "failed"
    cancelled  = "cancelled"

class SourceType(str, Enum):
    url  = "url"
    file = "file"

class Tone(str, Enum):
    formal = "formal"
    casual = "casual"

class HealthState(str, Enum):
    ok       = "ok"
    degraded = "degraded"
    down     = "down"

class ErrorCode(str, Enum):
    invalid_request        = "invalid_request"
    unauthorized           = "unauthorized"
    forbidden              = "forbidden"
    not_found              = "not_found"
    namespace_not_found    = "namespace_not_found"
    duplicate_job          = "duplicate_job"
    payload_too_large      = "payload_too_large"
    unsupported_media_type = "unsupported_media_type"
    validation_error       = "validation_error"
    rate_limited           = "rate_limited"
    internal_error         = "internal_error"
    upstream_error         = "upstream_error"
    service_unavailable    = "service_unavailable"
    timeout                = "timeout"

class ChunkingStrategy(str, Enum):
    auto      = "auto"       # detect from text structure
    heading   = "heading"    # numbered sections, markdown headings
    paragraph = "paragraph"  # blank-line separated blocks
    sentence  = "sentence"   # sentence boundaries (.!?)
    page      = "page"       # one chunk per PDF page
    sliding   = "sliding"    # classic sliding window fallback
    custom    = "custom"     # user-defined split_on separators

# ── Common ─────────────────────────────────────────────────────────────────────

class Usage(BaseModel):
    input_tokens:  int   = Field(..., ge=0)
    output_tokens: int   = Field(..., ge=0)
    cost_usd:      float = Field(..., ge=0.0)
    model_id:      str


class Chunk(BaseModel):
    chunk_id:     str            = Field(...)
    content:      str            = Field(..., max_length=4000)
    page_number:  int | None     = Field(None, ge=1)
    source_id:    str            = Field(...)
    source_url:   str | None     = Field(None)
    source_title: str | None     = Field(None)
    namespace_id: str            = Field(...)
    score:        float          = Field(..., ge=0.0, le=1.0)
    metadata:     dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    marker: str   = Field(...)
    chunk:  Chunk

# ── Query ──────────────────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str


class StyleHints(BaseModel):
    answer_max_chars: int  = Field(default=2000, ge=100, le=10000)
    cite_inline:      bool = Field(default=True)
    tone:             Tone = Field(default=Tone.formal)


class QueryRequest(BaseModel):
    question:             str                     = Field(..., max_length=2000)
    language:             str | None              = Field(None, description="ISO 639-1 code, e.g. 'en', 'ro', 'de'. Auto-detected if omitted.")
    namespaces:           list[str]               = Field(..., min_length=1, max_length=10)
    top_k:                int                     = Field(default=10, ge=1, le=50)
    hint_page_number:     int | None              = Field(None, description="Restrict initial search to a specific PDF page number.")
    rerank:               bool                    = Field(default=True)
    include_answer:       bool                    = Field(default=True)
    conversation_history: list[ConversationTurn]  = Field(default_factory=list, max_length=15)
    style_hints:          StyleHints | None       = Field(None)

    @field_validator("namespaces", mode="before")
    @classmethod
    def _validate_namespace_ids(cls, v: list) -> list:
        import re
        for ns in v:
            if not re.match(_NS_ID_PATTERN, str(ns)):
                raise ValueError(
                    f"namespace_id '{ns}' is invalid. "
                    "Must start with alphanumeric and contain only [a-zA-Z0-9_.-], max 256 chars."
                )
        return v


class QueryResponse(BaseModel):
    request_id:         str            = Field(...)
    answer:             str | None     = Field(None)
    citations:          list[Citation] = Field(default_factory=list)
    usage:              Usage
    latency_ms:         int            = Field(..., ge=0)
    model_version:      str
    retrieval_strategy: str | None     = Field(None)
    confidence:         float | None   = Field(None, ge=0.0, le=1.0)
    trace_id:           str | None     = Field(None)

# ── Ingest ─────────────────────────────────────────────────────────────────────

class IngestMetadata(BaseModel):
    source_title:  str | None = None
    language:      str | None = None
    published_at:  str | None = None
    model_config = {"extra": "allow"}


class ChunkingConfig(BaseModel):
    strategy:       ChunkingStrategy = Field(default=ChunkingStrategy.auto)
    chunk_size:     int              = Field(default=800, ge=100, le=4000)
    overlap:        int              = Field(default=100, ge=0, le=500)
    split_on:       list[str]        = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "],
        description="Ordered separator list for 'custom' strategy.",
    )
    min_chunk_size: int              = Field(default=50, ge=0, le=500)


class IngestRequest(BaseModel):
    namespace_id:   str                   = Field(..., pattern=_NS_ID_PATTERN, max_length=256)
    source_id:      str                   = Field(..., max_length=512)
    source_type:    SourceType            = Field(...)
    url:            str | None            = Field(None)
    mime_type_hint: str | None            = Field(None)
    language:       str | None            = Field(None, description="ISO 639-1 code. Auto-detected from content if omitted.")
    chunking:       ChunkingConfig        = Field(default_factory=ChunkingConfig)
    metadata:       IngestMetadata | None = Field(None)
    callback_url:   str | None            = Field(None)
    incremental:    bool                  = Field(
        default=False,
        description="If True, skip deleting existing chunks before indexing (append-only). "
                    "Use when adding new sources to an existing namespace without re-indexing."
    )

    @model_validator(mode="after")
    def url_required_for_url_type(self) -> IngestRequest:
        if self.source_type == SourceType.url and not self.url:
            raise ValueError("'url' is required when source_type='url'")
        return self


class IngestProgress(BaseModel):
    stage:          IngestStatus = Field(...)
    percent:        int          = Field(..., ge=0, le=100)
    chunks_created: int          = Field(..., ge=0)


class IngestError(BaseModel):
    code:      str
    message:   str
    retryable: bool


class IngestJob(BaseModel):
    job_id:                  str                  = Field(...)
    namespace_id:            str | None           = Field(None)
    source_id:               str | None           = Field(None)
    status:                  IngestStatus
    progress:                IngestProgress | None = Field(None)
    submitted_at:            str                  = Field(...)
    completed_at:            str | None           = Field(None)
    estimated_completion_at: str | None           = Field(None)
    error:                   IngestError | None   = Field(None)
    callback_status:         str | None           = Field(
        None,
        description="Delivery status of the callback_url webhook: 'sent', 'failed', or None if no callback_url was set."
    )

# ── Namespace ──────────────────────────────────────────────────────────────────

class NamespaceStats(BaseModel):
    namespace_id:         str        = Field(...)
    chunk_count:          int        = Field(..., ge=0)
    source_count:         int        = Field(..., ge=0)
    total_tokens_indexed: int        = Field(..., ge=0)
    last_ingested_at:     str | None = Field(None)
    embedding_model:      str        = Field(...)
    embedding_dim:        int        = Field(..., ge=1)


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthStatus(BaseModel):
    status:         HealthState      = Field(...)
    version:        str              = Field(...)
    uptime_seconds: int              = Field(..., ge=0)
    dependencies:   dict[str, str]   = Field(default_factory=dict)


# ── Errors ─────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code:       ErrorCode
    message:    str
    request_id: str
    details:    dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ── Delete job ─────────────────────────────────────────────────────────────────

class DeleteJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    sla:    str = "24h"


# ── Batch query (Session D) ────────────────────────────────────────────────────

class BatchQueryRequest(BaseModel):
    """POST /v1/query/batch — run up to BATCH_QUERY_MAX queries in parallel."""
    queries: list[QueryRequest] = Field(..., min_length=1, max_length=10)


class BatchQueryResponse(BaseModel):
    """Results ordered to match the input queries list (same index)."""
    results: list[QueryResponse | dict]  # dict used for per-query error entries