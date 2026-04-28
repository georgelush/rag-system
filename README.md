# RAG Service

A standalone, production-ready **Retrieval-Augmented Generation (RAG)** API service. Deploy it once; any caller — LLM agent, web application, automation pipeline — can index documents and ask natural-language questions against them over a standard HTTP/JSON API.

**Stack:** FastAPI · Qdrant (hybrid dense+sparse search) · Redis (queue, cache, rate limiting) · OpenAI-compatible embeddings & Whisper · Langfuse observability · lingua language detection

---

## Architecture

Two independent pipelines share the same vector store.

### Ingest pipeline — `POST /v1/ingest`

```
[caller] ──POST /v1/ingest──► [RAG service]
  1. validate + idempotency check (Idempotency-Key header)
  2. enqueue job → Redis list (rag:queue:ingest)
  returns job_id immediately (HTTP 202)

[worker] ──BRPOP rag:queue:ingest──►
  1. fetch_content   — download URL or receive file bytes
  2. extract         — convert to plain text (13 formats)
  3. detect_language — lingua, 75 languages, local
  4. chunk           — pick strategy (auto or caller-specified)
  5. embed (dense)   — OpenAI embeddings API, batches of EMBED_BATCH_SIZE=500
     encode (sparse) — TF-based sparse vectors, local, no API call
  6. upsert          — store dense + sparse vectors in Qdrant

poll status → GET /v1/ingest/{job_id}
```

### Query pipeline — `POST /v1/query`

```
[caller] ──POST /v1/query──► [RAG service]
  1. cache check     — skip pipeline entirely on hit
  2. detect_language — lingua or caller-supplied code
  3. embed_question  — same embedding model as ingest (dense)
     encode_sparse   — TF sparse vector (local)
  4. hybrid search   — RRF fusion (dense + sparse) in Qdrant
                       falls back to dense-only for legacy collections
                       page-boosted if hint_page_number is set
  5. rerank          — optional cross-encoder sidecar (RERANKER_URL)
  6. build_prompt    — context as [1]: … [2]: … markers
                       system prompt in detected language
  7. LLM completion  — gpt-4o-mini or LLM_MODEL
  8. Langfuse trace  — spans per stage, token usage, cost_usd

returns answer + citations + usage + latency + retrieval_strategy
```

### Tenant isolation

Each `(X-Tenant-ID, namespace_id)` pair maps to a dedicated Qdrant collection (`{tenant_id}__{namespace_id}`). Tenant identity comes from the request header, never the body.

### Worker process

The API enqueues jobs and returns immediately. A separate worker (`rag/worker.py`) consumes the queue:

- Configurable concurrency via `WORKER_CONCURRENCY` (default 4)
- Graceful shutdown on `SIGINT`/`SIGTERM` — drains in-flight jobs before exiting
- Failed jobs retried with exponential backoff; after `WORKER_MAX_RETRIES` → dead-letter queue (`rag:queue:dlq`)
- Run standalone: `python rag/worker.py`

---

## Features

### Document Ingest

| Capability | Detail |
|---|---|
| Source types | URL (HTTP/HTTPS) or direct file upload (multipart) |
| Idempotency | `Idempotency-Key` header deduplicates re-submissions |
| Re-indexing | Submitting the same `source_id` deletes existing chunks first (unless `incremental: true`) |
| Incremental | `incremental: true` appends new chunks without deleting existing ones |
| Async jobs | Returns `job_id` immediately; poll `GET /v1/ingest/{job_id}` |
| Job states | `queued → fetching → extracting → chunking → embedding → indexing → done / failed` |
| Job TTL | State stored in Redis for 7 days |
| Language | Auto-detected from content; overridable per request |
| Callbacks | `callback_url` called on `done` or `failed` with optional HMAC-SHA256 signature |

### Supported File Formats

| Format | Extension | Library |
|---|---|---|
| PDF | `.pdf` | pypdf |
| HTML | `.html`, `.htm` | regex tag-strip |
| Plain text | `.txt` | — |
| Markdown | `.md` | treated as plain text |
| Word | `.docx` | python-docx |
| PowerPoint | `.pptx` | python-pptx |
| Excel | `.xlsx` | openpyxl |
| CSV | `.csv` | stdlib |
| JSON | `.json` | serialized to readable text |
| JSONL | `.jsonl` | one object per line |
| XML | `.xml` | ElementTree tag-strip |
| EPUB | `.epub` | OPF/HTML chapters from ZIP |
| Audio | `.mp3`, `.wav`, `.m4a`, … | OpenAI Whisper |

MIME type is auto-detected from content bytes; `mime_type_hint` overrides.

### Chunking Strategies

| Strategy | Description |
|---|---|
| `auto` | Inspects headings, blank-line density, sentence patterns to pick the best strategy |
| `heading` | Splits on numbered sections or Markdown headings |
| `paragraph` | Splits on blank lines |
| `sentence` | Splits on `.`, `!`, `?` boundaries |
| `page` | One chunk per PDF page; preserves page number metadata |
| `sliding` | Classic sliding window with configurable overlap |
| `custom` | Caller supplies an ordered `split_on` separator list |

**Chunking parameters (`ChunkingConfig`):**

| Field | Default | Range | Description |
|---|---|---|---|
| `strategy` | `auto` | — | Strategy name |
| `chunk_size` | `800` | 100–4000 | Target chunk length in characters |
| `overlap` | `100` | 0–500 | Character overlap between consecutive sliding chunks |
| `split_on` | `["\n\n","\n",". "," "]` | — | Separator list for `custom` strategy |
| `min_chunk_size` | `50` | 0–500 | Chunks below this size are discarded |

### Hybrid Search

- **Dense vectors:** OpenAI `text-embedding-3-small` (1536-dim) or `text-embedding-3-large` (3072-dim)
- **Sparse vectors:** local TF encoder (`rag/sparse_encoder.py`) — tokenize → term frequency → 24-bit MD5 hash indices. No external API call.
- **Fusion:** Qdrant Reciprocal Rank Fusion (RRF) merges both ranked lists
- **Backward compatible:** dense-only collections (pre-hybrid ingest) use the legacy dense-only search path
- `retrieval_strategy` in the response reflects the actual method: `hybrid`, `hybrid+page_boost`, `semantic`, `semantic+page_boost`, and any of those with a `+reranked` suffix

### Language Detection

- Library: **lingua 2.0** — local, no external API call
- Supports 75 languages, returns ISO 639-1 codes (`"en"`, `"de"`, `"fr"`, …)
- Used at ingest (stored in metadata) and at query (selects system prompt language)
- Overridable by the caller on both endpoints

### Answer Generation

- Model: `LLM_MODEL` (default `gpt-4o-mini`)
- Numbered citation markers `[1]: content`, `[2]: content`, … in context
- System prompt language matches detected or declared question language
- Conversation history: up to 15 prior turns
- Style hints: `tone` (`formal`/`casual`), `cite_inline` (bool), `answer_max_chars` (100–10000)
- `cost_usd` computed from provider pricing tables

### Cross-Encoder Reranking

Optional sidecar (`reranker/`) reorders retrieved chunks by cross-encoder score:

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (configurable via `RERANKER_MODEL`)
- Activated when `RERANKER_URL` is set **and** `rerank: true` in the request (default)
- Fails silently on timeout or network error — query continues with score-sorted order
- PyTorch and sentence-transformers are kept out of the main image

### Query Result Cache

Identical queries skip the embed → search → LLM pipeline entirely:

- Cache key: `SHA-256(question + sorted namespaces + top_k)` per tenant
- TTL: `QUERY_CACHE_TTL_S` (default 5 min)
- Auto-invalidated when any source in the namespace is re-ingested
- Bypassed when `include_answer: false` or for streaming requests

### Rate Limiting

Per-tenant fixed-window limiting (Redis `INCR`):

| Endpoint group | Default |
|---|---|
| `POST /v1/query` | 60 req/min per tenant |
| `POST /v1/ingest` | 20 req/min per tenant |
| All other endpoints | 120 req/min per tenant |

Returns `429 Too Many Requests` with a `Retry-After` header. Fails open if Redis is unavailable.

### Prometheus Metrics

`GET /metrics` exposes Prometheus text-format metrics:

| Metric | Type |
|---|---|
| `rag_http_requests_total` | Counter (`method`, `path`, `status_code`) |
| `rag_http_request_duration_seconds` | Histogram (`method`, `path`) |
| `rag_ingest_queue_depth` | Gauge |
| `rag_ingest_dlq_depth` | Gauge |
| `rag_cache_hits_total` | Counter |
| `rag_cache_misses_total` | Counter |
| `rag_reranker_failures_total` | Counter |
| `rag_ingest_retries_total` | Counter |
| `rag_namespace_queries_total` | Counter (`namespace_id`) |

Disable with `METRICS_ENABLED=false`.

### Namespace-level RBAC

When `RBAC_ENABLED=true` and `RBAC_SECRET` is set:

- Mint a scoped JWT via `POST /v1/tokens` (master API key required)
- Use as `Authorization: Bearer <token>` on any endpoint
- Tokens carry per-namespace permissions: `read`, `write`, `admin`; use `"*"` to grant all namespaces

**All endpoints and required access:**

| Method | Path | Auth required | RBAC permission |
|---|---|---|---|
| `GET` | `/v1/health` | No | — |
| `GET` | `/metrics` | No | — |
| `GET` | `/v1/openapi.json` | No | — |
| `GET` | `/v1/ingest/{job_id}` | Yes (any valid token) | none — no namespace check |
| `POST` | `/v1/query` | Yes | `read` on each namespace in request |
| `POST` | `/v1/query/batch` | Yes | `read` on each namespace in each query |
| `POST` | `/v1/query/stream` | Yes | `read` on each namespace in request |
| `GET` | `/v1/namespaces/{ns}/stats` | Yes | `read` on `{ns}` |
| `POST` | `/v1/ingest` | Yes | `write` on `namespace_id` in request body |
| `DELETE` | `/v1/namespaces/{ns}/sources/{src}` | Yes | `write` on `{ns}` |
| `DELETE` | `/v1/namespaces/{ns}` | Yes | `admin` on `{ns}` |
| `POST` | `/v1/tokens` | Yes (master key only) | — |

`RBAC_ENABLED=false` (default) — only the master API key is accepted; JWT tokens are not issued or verified.

### Structured Logging

All output is newline-delimited JSON:

```json
{"ts": "2026-04-27T10:00:00Z", "level": "INFO", "logger": "rag.query", "msg": "Query completed", "request_id": "...", "latency_ms": 420, "cost_usd": 0.00014}
```

Log level via `LOG_LEVEL` (default `INFO`).

### Ingest Reliability

- `WORKER_MAX_RETRIES` attempts with exponential backoff (`WORKER_RETRY_DELAY_S` base, doubles each retry)
- Non-retryable errors (`ValueError`, `NotImplementedError`, empty content, unsupported format) go directly to DLQ
- DLQ depth visible on `GET /metrics`

### Observability (Langfuse)

When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, every request is traced with spans per stage and token/cost attribution. Completely disabled (no-op) when keys are absent.

### Embedding Model Migration

```bash
python rag/migrate.py --tenant <tenant> --namespace <ns> \
  --from text-embedding-3-small --to text-embedding-3-large
```

Re-embeds all chunks in batches using a temp-collection pattern (atomic swap). Resumable if interrupted. `--dry-run` prints chunk count and target dimensions without modifying anything.

---

## API Reference

Live schema: `GET /v1/openapi.json`

### Authentication

Every endpoint except `GET /v1/health`, `GET /metrics`, and `GET /v1/openapi.json` requires:

```
Authorization: Bearer <RAG_API_KEY>
X-Request-ID:  <any-unique-string>
X-Tenant-ID:   <tenant-identifier>
```

| Violation | HTTP | Error code |
|---|---|---|
| Missing/wrong `Authorization` | `401` | `unauthorized` |
| Missing `X-Request-ID` | `400` | `invalid_request` |
| Missing `X-Tenant-ID` | `400` | `invalid_request` |

### Endpoint Summary

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/health` | Liveness + dependency status (no auth) |
| `POST` | `/v1/ingest` | Submit document for async indexing |
| `GET` | `/v1/ingest/{job_id}` | Poll ingest job status |
| `POST` | `/v1/query` | Ask a question, receive answer + citations |
| `POST` | `/v1/query/batch` | Up to `BATCH_QUERY_MAX` queries in parallel |
| `POST` | `/v1/query/stream` | Streaming answer via Server-Sent Events |
| `POST` | `/v1/tokens` | Mint a scoped namespace JWT (master key only) |
| `GET` | `/v1/namespaces/{ns}/stats` | Chunk count, source count, last indexed timestamp |
| `DELETE` | `/v1/namespaces/{ns}/sources/{src}` | Delete a source and all its chunks |
| `DELETE` | `/v1/namespaces/{ns}` | Drop entire namespace (async) |
| `GET` | `/metrics` | Prometheus text-format metrics |
| `GET` | `/v1/openapi.json` | OpenAPI schema |

---

### `GET /v1/health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "dependencies": { "qdrant": "ok", "redis": "ok" }
}
```

Returns `503` with `status: "down"` if either dependency is unreachable.

---

### `POST /v1/ingest`

Requires `Idempotency-Key` header. Returns `202` immediately with a `job_id`; the actual processing happens asynchronously in the worker.

---

#### Required fields (always)

| Field | Type | Description |
|---|---|---|
| `namespace_id` | string | Target namespace (alphanumeric, `_`, `.`, `-`, max 256 chars) |
| `source_id` | string | Logical identifier for this document (max 512 chars) |
| `source_type` | `"url"` or `"file"` | How the content is delivered |

If `source_type = "url"`, the field `url` is also required.

Everything else is **optional** — omit any field and the service picks a sensible default.

---

#### Example 1 — Minimum, ingest from URL

```http
POST /v1/ingest
Authorization: Bearer <RAG_API_KEY>
X-Request-ID: req-001
X-Tenant-ID: acme
Idempotency-Key: key-001
Content-Type: application/json

{
  "namespace_id": "my-docs",
  "source_id":    "report-q4",
  "source_type":  "url",
  "url":          "https://example.com/report.pdf"
}
```

The service auto-detects language, picks the best chunking strategy, uses default chunk size (800 chars), and stores no extra metadata.

---

#### Example 2 — Minimum, upload a file

File uploads use `multipart/form-data`. The JSON payload goes in a form field named **`payload`** (as a JSON string); the file bytes go in a form field named **`file`**.

```http
POST /v1/ingest
Authorization: Bearer <RAG_API_KEY>
X-Request-ID: req-002
X-Tenant-ID: acme
Idempotency-Key: key-002
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="payload"

{"namespace_id":"my-docs","source_id":"contract-2024","source_type":"file"}
--boundary
Content-Disposition: form-data; name="file"; filename="contract.pdf"
Content-Type: application/pdf

<binary file bytes>
--boundary--
```

Using curl:

```bash
curl -X POST http://localhost:8080/v1/ingest \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "X-Request-ID: req-002" \
  -H "X-Tenant-ID: acme" \
  -H "Idempotency-Key: key-002" \
  -F 'payload={"namespace_id":"my-docs","source_id":"contract-2024","source_type":"file"}' \
  -F "file=@contract.pdf"
```

---

#### Example 3 — Full control (all optional fields set)

```json
{
  "namespace_id":  "my-docs",
  "source_id":     "contract-2024",
  "source_type":   "file",

  "language":      "ro",

  "chunking": {
    "strategy":       "sliding",
    "chunk_size":     500,
    "overlap":        80,
    "min_chunk_size": 30
  },

  "metadata": {
    "source_title":  "Contract de prestări servicii 2024",
    "document_type": "contract",
    "published_at":  "2024-01-15"
  },

  "callback_url":  "https://my-app.example.com/webhooks/ingest",
  "incremental":   false,
  "mime_type_hint": "application/pdf"
}
```

---

#### All optional fields

| Field | Default | Description |
|---|---|---|
| `language` | auto-detect | ISO 639-1 code (`"en"`, `"ro"`, `"de"`, …). Skip detection and force a language. |
| `mime_type_hint` | auto-detect | Force MIME type instead of detecting from file bytes. |
| `incremental` | `false` | `true` = append new chunks without deleting existing ones for this `source_id`. `false` = delete all existing chunks for this `source_id` first, then re-index. |
| `callback_url` | null | URL to call (HTTP POST) when the job reaches `done` or `failed`. |
| `metadata` | null | Descriptive fields stored alongside each chunk (not used for search). |
| `chunking` | strategy `auto` | How to split the document into chunks. See below. |

**`metadata` sub-fields** (all optional, all strings):

| Field | Description |
|---|---|
| `source_title` | Human-readable title displayed in citations |
| `document_type` | Free label, e.g. `"contract"`, `"report"`, `"manual"` |
| `published_at` | ISO 8601 date string |
| `language` | Stored as metadata; does not override extraction language detection |
| _(any extra key)_ | Additional custom fields — stored as-is |

**`chunking` sub-fields:** see [Chunking Strategies](#chunking-strategies) in the Features section for all strategies, parameter defaults, and ranges.

> **Mix and match:** You can set only the fields you care about. For example, keep `strategy: auto` but override `chunk_size: 400` and `min_chunk_size: 100` — the rest use defaults.

---

#### Valid values for each field

**`language`** — ISO 639-1 codes accepted:

| Code | Language | Code | Language | Code | Language |
|---|---|---|---|---|---|
| `af` | Afrikaans | `fr` | French | `pt` | Portuguese |
| `ar` | Arabic | `ga` | Irish | `ro` | Romanian |
| `az` | Azerbaijani | `gl` | Galician | `ru` | Russian |
| `be` | Belarusian | `gu` | Gujarati | `sk` | Slovak |
| `bg` | Bulgarian | `he` | Hebrew | `sl` | Slovenian |
| `bn` | Bengali | `hi` | Hindi | `sq` | Albanian |
| `bs` | Bosnian | `hr` | Croatian | `sr` | Serbian |
| `ca` | Catalan | `hu` | Hungarian | `sv` | Swedish |
| `cs` | Czech | `hy` | Armenian | `ta` | Tamil |
| `cy` | Welsh | `id` | Indonesian | `te` | Telugu |
| `da` | Danish | `is` | Icelandic | `th` | Thai |
| `de` | German | `it` | Italian | `tl` | Filipino |
| `el` | Greek | `ja` | Japanese | `tr` | Turkish |
| `en` | English | `ka` | Georgian | `uk` | Ukrainian |
| `eo` | Esperanto | `kk` | Kazakh | `ur` | Urdu |
| `es` | Spanish | `ko` | Korean | `uz` | Uzbek |
| `et` | Estonian | `la` | Latin | `vi` | Vietnamese |
| `eu` | Basque | `lt` | Lithuanian | `zh` | Chinese |
| `fa` | Persian | `lv` | Latvian | | |
| `fi` | Finnish | `mk` | Macedonian | | |
| `mn` | Mongolian | `mr` | Marathi | | |
| `ms` | Malay | `nb` | Norwegian Bokmål | | |
| `nl` | Dutch | `nn` | Norwegian Nynorsk | | |
| `pa` | Punjabi | `pl` | Polish | | |

If omitted, lingua auto-detects from content (works best on ≥ 10 words). Falls back to `"en"` if confidence is too low.

---

**`mime_type_hint`** — accepted values:

| Value | Format |
|---|---|
| `application/pdf` | PDF |
| `text/html` | HTML |
| `text/plain` | TXT |
| `text/markdown` | Markdown |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | DOCX |
| `application/vnd.openxmlformats-officedocument.presentationml.presentation` | PPTX |
| `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` | XLSX |
| `text/csv` | CSV |
| `application/json` | JSON |
| `application/jsonlines` | JSONL |
| `application/xml` | XML |
| `application/epub+zip` | EPUB |
| `audio/mpeg` | MP3 |
| `audio/mp4` | MP4 audio |
| `audio/wav` | WAV |
| `audio/ogg` | OGG |
| `audio/flac` | FLAC |
| `video/webm` | WEBM |
| `audio/m4a` | M4A |

If omitted, MIME is auto-detected from file extension first, then from magic bytes in the file content.

---

**`source_type`** — exactly two values:

| Value | When to use |
|---|---|
| `"url"` | The content lives at an HTTP/HTTPS URL. Must also set `url`. |
| `"file"` | The content is uploaded directly as a file in the multipart body. |

---

**`incremental`** — boolean:

| Value | Behaviour |
|---|---|
| `false` (default) | Deletes all existing chunks for this `source_id` first, then indexes the new version. Use for updates/replacements. |
| `true` | Keeps existing chunks for this `source_id` and adds the new ones on top. Use when adding new documents to a namespace without re-indexing old ones. |

---

**Response `202`:**
```json
{ "job_id": "550e8400-...", "status": "queued", "submitted_at": "2026-04-27T10:00:00Z" }
```

---

### `GET /v1/ingest/{job_id}`

```json
{
  "job_id": "550e8400-...",
  "status": "done",
  "progress": { "stage": "done", "percent": 100, "chunks_created": 42 },
  "submitted_at": "2026-04-27T10:00:00Z",
  "completed_at": "2026-04-27T10:00:15Z",
  "callback_status": "sent"
}
```

`callback_status` is `"sent"`, `"failed"`, or `null` (no callback configured).

---

### `POST /v1/query`

```json
{
  "question":   "What are the main findings?",
  "namespaces": ["my-docs"],
  "top_k":      5,
  "language":   "en",
  "hint_page_number": 3,
  "rerank":     true,
  "include_answer": true,
  "conversation_history": [
    { "role": "user",      "content": "Tell me about Q3." },
    { "role": "assistant", "content": "Q3 showed 12% growth..." }
  ],
  "style_hints": { "tone": "formal", "cite_inline": true, "answer_max_chars": 2000 }
}
```

| Field | Required | Default |
|---|---|---|
| `question` | yes | — |
| `namespaces` | yes | — |
| `language` | no | auto-detect |
| `top_k` | no | `10` |
| `hint_page_number` | no | null |
| `rerank` | no | `true` |
| `include_answer` | no | `true` |
| `conversation_history` | no | `[]` |
| `style_hints` | no | defaults |

**Response `200`:**
```json
{
  "request_id": "req-abc",
  "answer": "The Q4 report highlights a 15% revenue increase [1].",
  "citations": [
    {
      "marker": "[1]",
      "chunk": {
        "chunk_id":    "uuid",
        "content":     "Revenue increased 15% year-over-year...",
        "page_number": 3,
        "source_id":   "report-2024-q4",
        "score":       0.92,
        "namespace_id": "my-docs"
      }
    }
  ],
  "usage": { "input_tokens": 312, "output_tokens": 88, "cost_usd": 0.00014, "model_id": "gpt-4o-mini" },
  "latency_ms": 1240,
  "model_version": "gpt-4o-mini",
  "retrieval_strategy": "hybrid+reranked"
}
```

---

### `POST /v1/query/batch`

Run up to `BATCH_QUERY_MAX` (default 10) independent queries in parallel. Results are returned in input order. A single query failure does not abort the batch — the failed entry is returned inline as `{ "error": { "code": "...", "message": "..." } }`.

```json
{
  "queries": [
    { "question": "What are the key findings?", "namespaces": ["docs"] },
    { "question": "Who authored the report?",   "namespaces": ["docs"] }
  ]
}
```

Returns `422` if `queries` is empty or exceeds `BATCH_QUERY_MAX`.

---

### `POST /v1/query/stream`

Identical request body to `POST /v1/query`. Returns `text/event-stream` (Server-Sent Events).

| Event | When | Data |
|---|---|---|
| `citations` | After retrieval, before LLM | JSON array of Citation objects |
| `token` | Once per LLM token | `{"token": "..."}` |
| `done` | After last token | `{"request_id", "retrieval_strategy", "usage", "latency_ms"}` |
| `error` | On failure | `{"code", "message"}` |

Query result cache is **not** used for streaming responses.

---

### `GET /v1/namespaces/{namespace_id}/stats`

```json
{
  "namespace_id":   "my-docs",
  "tenant_id":      "my-tenant",
  "chunk_count":    420,
  "source_count":   10,
  "last_indexed_at": "2026-04-27T09:30:00Z"
}
```

Returns `404` if the namespace does not exist. Response is cached in Redis for `STATS_CACHE_TTL_S` seconds.

---

### `DELETE /v1/namespaces/{namespace_id}/sources/{source_id}`

Deletes all chunks for the given source. Returns `204 No Content`.

---

### `DELETE /v1/namespaces/{namespace_id}`

Schedules full namespace deletion. Returns `202 Accepted` with a `job_id`.

---

### `POST /v1/tokens`

Requires master API key. Mints a scoped namespace JWT.

```json
{
  "namespace_permissions": {
    "my-docs": ["read", "write"],
    "*":       ["read"]
  },
  "expires_in_seconds": 3600
}
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the required values.

| Variable | Required | Default | Description |
|---|---|---|---|
| `RAG_API_KEY` | **yes** | — | Bearer token all callers must send |
| `LLM_API_KEY` | **yes** | — | API key for the embedding/LLM provider |
| `LLM_PROXY` | no | — | Base URL of LiteLLM or any OpenAI-compatible proxy |
| `LLM_MODEL` | no | `gpt-4o-mini` | Model for answer generation |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` | `text-embedding-3-small` (1536-dim) or `text-embedding-3-large` (3072-dim). **Do not change after first ingest.** |
| `QDRANT_URL` | no | `http://localhost:6333` | Qdrant HTTP endpoint |
| `REDIS_URL` | no | `redis://localhost:6379` | Redis connection URL |
| `RAG_PORT` | no | `8080` | Listening port |
| `RAG_VERSION` | no | `1.0.0` | Version string in `/v1/health` |
| `LOG_LEVEL` | no | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LANGFUSE_PUBLIC_KEY` | no | — | Leave blank to disable tracing |
| `LANGFUSE_SECRET_KEY` | no | — | Must be set alongside public key |
| `LANGFUSE_HOST` | no | `https://cloud.langfuse.com` | Langfuse endpoint |
| `RATE_LIMIT_QUERY` | no | `60` | Max query requests per minute per tenant |
| `RATE_LIMIT_INGEST` | no | `20` | Max ingest requests per minute per tenant |
| `RATE_LIMIT_DEFAULT` | no | `120` | Default limit for all other endpoints |
| `WORKER_CONCURRENCY` | no | `4` | Parallel ingest jobs in the worker |
| `WORKER_MAX_RETRIES` | no | `3` | Max retries per failed job before DLQ |
| `WORKER_RETRY_DELAY_S` | no | `5` | Base backoff delay (doubles each retry) |
| `WORKER_SHUTDOWN_TIMEOUT_S` | no | `30` | Seconds to wait for in-flight jobs on shutdown |
| `WHISPER_MODEL` | no | `whisper-1` | OpenAI Whisper model for audio |
| `QUERY_CACHE_ENABLED` | no | `true` | Enable Redis query result cache |
| `QUERY_CACHE_TTL_S` | no | `300` | Cache TTL in seconds |
| `METRICS_ENABLED` | no | `true` | Expose `GET /metrics` |
| `WEBHOOK_SECRET` | no | — | HMAC-SHA256 secret for callback signatures |
| `RERANKER_URL` | no | — | URL of reranker sidecar; leave blank to disable |
| `RERANKER_TIMEOUT_S` | no | `10.0` | HTTP timeout for reranker sidecar calls |
| `RBAC_ENABLED` | no | `false` | Enable namespace-scoped JWT tokens |
| `RBAC_SECRET` | no | — | HS256 signing secret; required when `RBAC_ENABLED=true` |
| `EMBED_BATCH_SIZE` | no | `500` | Chunks per `embeddings.create()` call (max 2048) |
| `STATS_CACHE_TTL_S` | no | `60` | TTL for `/v1/namespaces/{id}/stats` cache; `0` to disable |
| `BATCH_QUERY_MAX` | no | `10` | Maximum parallel queries in `POST /v1/query/batch` |

The service refuses to start if `RAG_API_KEY` is missing. Missing optional secrets (`WEBHOOK_SECRET`, partial Langfuse keys, `RBAC_SECRET` when RBAC is enabled) are logged as `WARNING` at startup.

---

## Running

### Full stack (recommended)

```bash
cp .env.example .env
# fill in RAG_API_KEY, LLM_API_KEY
docker compose up -d
curl http://localhost:8080/v1/health
```

This starts: `rag` (API), `worker` (ingest queue consumer), `qdrant`, `redis`.

### Local development

```bash
# Start infrastructure only
docker compose -f compose.infra.yml up -d

# Python environment (3.12+)
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt

# API server
python rag_server.py

# Worker (separate terminal)
python rag/worker.py
```

### Reranker sidecar (optional)

```bash
docker compose up reranker
# set RERANKER_URL=http://reranker:8090 in .env to activate
```

### Docker build

```bash
docker build -t rag-service:1.0.0 .
docker compose up -d
```

---

## Testing

```bash
# All tests
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=rag --cov-report=term-missing
```

| Module | Test file |
|---|---|
| `rag/auth.py` | `tests/test_auth.py` |
| `rag/cache.py` | `tests/test_cache.py` |
| `rag/ingest.py` (callbacks) | `tests/test_callback.py` |
| `rag/chunkers/` | `tests/test_chunkers.py` |
| `rag/config.py` | `tests/test_config.py` |
| `rag/extractor.py` | `tests/test_extractor.py` |
| `rag/routers/health_router.py` | `tests/test_health.py` |
| `rag/ingest.py` (incremental) | `tests/test_incremental.py` |
| `rag/ingest.py` | `tests/test_ingest.py` |
| `rag/metrics.py` | `tests/test_metrics.py` |
| `rag/migrate.py` | `tests/test_migrate.py` |
| `rag/routers/namespaces_router.py` | `tests/test_namespaces.py` |
| `rag/query.py` | `tests/test_query.py` |
| `rag/routers/query_router.py` (batch) | `tests/test_batch_query.py` |
| `rag/rate_limiter.py` | `tests/test_rate_limiter.py` |
| `rag/rbac.py` | `tests/test_rbac.py` |
| reranker sidecar | `tests/test_reranker.py` |
| `rag/routers/stream_router.py` | `tests/test_stream.py` |
| `rag/vector_store.py` | `tests/test_vector_store.py` |
| `rag/worker.py` | `tests/test_worker.py` |

---

## Project Structure

```
rag-service/
├── rag_server.py                  # FastAPI app, lifespan, middleware wiring
├── rag/
│   ├── auth.py                    # Timing-safe bearer token + header validation
│   ├── cache.py                   # Redis query result cache (TTL + namespace invalidation)
│   ├── config.py                  # Centralised env-var config + startup validation
│   ├── cost.py                    # Token cost calculation (USD)
│   ├── extractor.py               # Text extraction from 13 file formats
│   ├── ingest.py                  # Ingest pipeline (fetch → chunk → embed → index)
│   ├── jobs.py                    # Job state CRUD in Redis (7-day TTL)
│   ├── language.py                # lingua language detection + ISO 639-1 utilities
│   ├── logging_config.py          # Structured JSON logging setup
│   ├── metrics.py                 # Prometheus metrics (counters, histogram, gauges)
│   ├── migrate.py                 # Embedding model migration CLI
│   ├── models.py                  # Pydantic v2 schemas (all request/response types)
│   ├── openai_client.py           # Singleton AsyncOpenAI client
│   ├── query.py                   # Query pipeline (embed → search → rerank → generate)
│   ├── queue.py                   # Redis list-based ingest queue (enqueue/dequeue/DLQ)
│   ├── rate_limiter.py            # Per-tenant fixed-window rate limiter (middleware)
│   ├── rbac.py                    # Namespace-scoped JWT auth (AuthResult, mint/decode)
│   ├── sparse_encoder.py          # Local TF sparse vector encoder
│   ├── telemetry.py               # Langfuse observability client (no-op if keys absent)
│   ├── vector_store.py            # Qdrant operations (hybrid upsert, RRF search, stats)
│   ├── worker.py                  # Asyncio ingest worker (BRPOP + graceful shutdown)
│   ├── chunkers/
│   │   ├── __init__.py            # Chunker registry + auto-detection
│   │   ├── base.py                # BaseChunker ABC
│   │   ├── custom.py              # Custom separator chunker
│   │   ├── heading.py             # Heading-boundary chunker
│   │   ├── page.py                # Page-boundary chunker (PDF)
│   │   ├── paragraph.py           # Blank-line chunker
│   │   ├── sentence.py            # Sentence-boundary chunker
│   │   └── sliding.py             # Sliding window chunker
│   └── routers/
│       ├── health_router.py       # GET /v1/health
│       ├── ingest_router.py       # POST /v1/ingest, GET /v1/ingest/{job_id}
│       ├── namespaces_router.py   # GET|DELETE /v1/namespaces/…
│       ├── query_router.py        # POST /v1/query, POST /v1/query/batch
│       ├── stream_router.py       # POST /v1/query/stream (SSE)
│       └── tokens_router.py       # POST /v1/tokens
├── reranker/
│   ├── app.py                     # Cross-encoder reranking microservice (FastAPI)
│   ├── requirements.txt           # sentence-transformers + fastapi
│   └── Dockerfile                 # PyTorch CPU-only image (~2 GB)
├── scripts/
│   └── fix_vector_store.py        # One-off vector store repair utility
├── tests/
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_batch_query.py
│   ├── test_cache.py
│   ├── test_callback.py
│   ├── test_chunkers.py
│   ├── test_config.py
│   ├── test_extractor.py
│   ├── test_health.py
│   ├── test_incremental.py
│   ├── test_ingest.py
│   ├── test_metrics.py
│   ├── test_migrate.py
│   ├── test_namespaces.py
│   ├── test_query.py
│   ├── test_rate_limiter.py
│   ├── test_rbac.py
│   ├── test_reranker.py
│   ├── test_stream.py
│   ├── test_vector_store.py
│   └── test_worker.py
├── Dockerfile
├── compose.yml                    # Full stack: rag + worker + qdrant + redis
├── compose.infra.yml              # Infrastructure only: qdrant + redis
├── requirements.txt               # Runtime + dev dependencies (pinned ranges)
├── pytest.ini                     # asyncio_mode=auto, testpaths=tests
└── .env.example                   # Environment variable template
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | `>=0.115.12,<1.0.0` | HTTP framework |
| `uvicorn` | `>=0.34.0,<1.0.0` | ASGI server |
| `pydantic` | `>=2.11.0,<3.0.0` | Schema validation |
| `qdrant-client` | `>=1.13.0,<2.0.0` | Vector store client |
| `redis` | `>=5.3.0,<6.0.0` | Job state, queue, cache, rate limiting |
| `openai` | `>=1.0.0,<3.0.0` | Embeddings, completions, Whisper |
| `httpx` | `>=0.28.0,<1.0.0` | Async HTTP (URL fetch) |
| `pypdf` | `>=5.3.0,<6.0.0` | PDF extraction |
| `python-docx` | `>=1.1.0,<2.0.0` | DOCX extraction |
| `python-pptx` | `>=1.0.0,<2.0.0` | PPTX extraction |
| `openpyxl` | `>=3.1.0,<4.0.0` | XLSX extraction |
| `lingua-language-detector` | `>=2.0.0,<3.0.0` | Language detection (75 languages) |
| `langfuse` | `>=2.0.0,<3.0.0` | Observability tracing (optional) |
| `prometheus-client` | `>=0.20.0,<1.0.0` | Prometheus metrics endpoint |
| `PyJWT` | `>=2.0.0,<3.0.0` | JWT minting and verification (RBAC) |
| `python-dotenv` | `>=1.1.0,<2.0.0` | `.env` loading |
| `python-multipart` | `>=0.0.20,<1.0.0` | File upload support |

---

## Known Limitations

- **Changing `EMBEDDING_MODEL` after first ingest** silently corrupts search (dimension mismatch). Use `python rag/migrate.py` to migrate an existing collection.
- **Sparse encoder uses TF, not BM25** — no IDF because corpus-level statistics are expensive to maintain per-namespace. TF alone still improves recall on keyword-heavy queries.
- **Language detection degrades on short texts** (< 20 words) — lingua quality drops without sufficient context.
- **Streaming responses are not cached** — SSE streams cannot be replayed from cache.
- **`POST /v1/tokens` requires master key** — JWT tokens cannot mint other tokens (privilege escalation prevention).

---

## License

