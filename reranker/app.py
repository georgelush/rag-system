"""
reranker/app.py — Cross-encoder reranking microservice.

Runs as a separate Docker service (see compose.yml).  The main RAG service
calls POST /rerank when ``rerank: true`` is set on a query request and
RERANKER_URL is configured.

POST /rerank
    Input:  {"query": str, "passages": [{"id": str, "text": str}]}
    Output: {"passages": [{"id": str, "score": float}]}  ← sorted descending by score

GET /health
    Returns {"status": "ok", "model": "<model-name>"}.

The cross-encoder model is loaded lazily on the first request so the service
starts quickly.  Set RERANKER_MODEL env var to change the model
(default: cross-encoder/ms-marco-MiniLM-L-6-v2).
"""

import logging
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger("reranker")

MODEL_NAME: str = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ── Lazy model singleton ──────────────────────────────────────────────────────

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder  # type: ignore
        log.info("Loading reranker model '%s'...", MODEL_NAME)
        _model = CrossEncoder(MODEL_NAME)
        log.info("Reranker model loaded.")
    return _model


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Passage(BaseModel):
    id: str
    text: str


class RerankRequest(BaseModel):
    query: str
    passages: list[Passage]


class ScoredPassage(BaseModel):
    id: str
    score: float


class RerankResponse(BaseModel):
    passages: list[ScoredPassage]


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Reranker", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.passages:
        return RerankResponse(passages=[])

    model = _get_model()
    pairs = [[req.query, p.text] for p in req.passages]
    raw_scores: list[float] = model.predict(pairs).tolist()

    scored = sorted(
        [
            ScoredPassage(id=p.id, score=float(s))
            for p, s in zip(req.passages, raw_scores)
        ],
        key=lambda x: x.score,
        reverse=True,
    )
    return RerankResponse(passages=scored)


if __name__ == "__main__":
    port = int(os.environ.get("RERANKER_PORT", "8090"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
