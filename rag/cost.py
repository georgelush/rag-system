"""
rag/cost.py — Token cost calculation for OpenAI-compatible models.

Prices as of April 2026. Update _EMBEDDING_COST_PER_1M and _LLM_COST_PER_1M
when provider pricing changes — nowhere else needs touching.

Supported embedding models (proxy-confirmed):
  text-embedding-3-small  — 1536 dims, $0.020/1M tokens
  text-embedding-3-large  — 3072 dims, $0.130/1M tokens
"""

# ── Embedding costs ($ per 1M input tokens) ───────────────────────────────────

_EMBEDDING_COST_PER_1M: dict[str, float] = {
    "text-embedding-3-small": 0.020,
    "text-embedding-3-large": 0.130,
}

# ── LLM costs ($ per 1M tokens) — (input_cost, output_cost) ──────────────────

_LLM_COST_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":            (0.150,  0.600),
    "gpt-4o-mini-2024-07-18": (0.150,  0.600),
    "gpt-4o":                 (2.500, 10.000),
    "gpt-4o-2024-11-20":      (2.500, 10.000),
    "gpt-4.1":                (2.000,  8.000),
    "gpt-4.1-mini":           (0.400,  1.600),
    "gpt-4.1-nano":           (0.100,  0.400),
    "gpt-5.4-nano":           (0.100,  0.400),  # proxy alias
}

_DEFAULT_EMBED_COST  = 0.020   # conservative fallback for unknown models
_DEFAULT_INPUT_COST  = 0.150
_DEFAULT_OUTPUT_COST = 0.600


def embedding_cost_usd(model: str, token_count: int) -> float:
    """Return USD cost for embedding `token_count` tokens with `model`."""
    cost_per_1m = _EMBEDDING_COST_PER_1M.get(model, _DEFAULT_EMBED_COST)
    return round((token_count / 1_000_000) * cost_per_1m, 8)


def llm_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost for an LLM completion call with `model`."""
    in_cost, out_cost = _LLM_COST_PER_1M.get(
        model, (_DEFAULT_INPUT_COST, _DEFAULT_OUTPUT_COST)
    )
    return round(
        (input_tokens  / 1_000_000) * in_cost
        + (output_tokens / 1_000_000) * out_cost,
        8,
    )
