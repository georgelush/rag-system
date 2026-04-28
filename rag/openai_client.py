"""
rag/openai_client.py — Shared AsyncOpenAI client singleton.

All modules (ingest, query, extractor) import from here to avoid
creating a new HTTP client per request.
"""

import os

from openai import AsyncOpenAI

_LLM_PROXY   = os.environ.get("LLM_PROXY", "")
_LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

openai_client = AsyncOpenAI(
    api_key=_LLM_API_KEY,
    base_url=f"{_LLM_PROXY}/v1" if _LLM_PROXY else None,
)

__all__ = ["openai_client"]
