"""
tests/test_cache.py — Unit tests for rag/cache.py query result cache.

Covers:
  - Cache hit → returns stored dict, increments CACHE_HITS
  - Cache miss → returns None, increments CACHE_MISSES
  - Cache disabled (QUERY_CACHE_ENABLED=false) → always miss, no Redis call
  - set_cached → stores under correct key, registers in namespace index
  - invalidate_namespace → deletes all keys in the namespace index
  - Redis unavailable → get/set/invalidate fail silently (no exception raised)
  - Cache bypass when include_answer=False (tested in run_query path)
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_redis(get_return=None) -> AsyncMock:
    r = AsyncMock()
    r.get = AsyncMock(return_value=get_return)
    r.set = AsyncMock(return_value=True)
    r.sadd = AsyncMock(return_value=1)
    r.expire = AsyncMock(return_value=True)
    r.smembers = AsyncMock(return_value=set())
    r.delete = AsyncMock(return_value=1)
    return r


_SAMPLE_RESPONSE = {
    "request_id": "req-1",
    "answer": "Cached answer.",
    "citations": [],
    "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.0, "model_id": "test"},
    "latency_ms": 100,
    "model_version": "test",
}


# ── get_cached ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_cached_returns_none_on_miss():
    from rag.cache import get_cached
    redis = _make_redis(get_return=None)
    result = await get_cached(redis, "tenant-1", "What?", ["ns-a"], 10)
    assert result is None


@pytest.mark.asyncio
async def test_get_cached_returns_dict_on_hit():
    from rag.cache import get_cached
    raw = json.dumps(_SAMPLE_RESPONSE).encode()
    redis = _make_redis(get_return=raw)
    result = await get_cached(redis, "tenant-1", "What?", ["ns-a"], 10)
    assert result is not None
    assert result["answer"] == "Cached answer."


@pytest.mark.asyncio
async def test_get_cached_hit_increments_counter():
    from rag.cache import get_cached
    from rag.metrics import CACHE_HITS
    raw = json.dumps(_SAMPLE_RESPONSE).encode()
    redis = _make_redis(get_return=raw)

    before = CACHE_HITS._value.get()
    await get_cached(redis, "tenant-1", "Q?", ["ns"], 5)
    assert CACHE_HITS._value.get() == before + 1


@pytest.mark.asyncio
async def test_get_cached_miss_increments_counter():
    from rag.cache import get_cached
    from rag.metrics import CACHE_MISSES
    redis = _make_redis(get_return=None)

    before = CACHE_MISSES._value.get()
    await get_cached(redis, "tenant-1", "Q?", ["ns"], 5)
    assert CACHE_MISSES._value.get() == before + 1


@pytest.mark.asyncio
async def test_get_cached_disabled_returns_none_without_redis_call():
    """QUERY_CACHE_ENABLED=false → returns None immediately, no Redis call."""
    import importlib
    import rag.cache as cache_mod

    redis = _make_redis()
    with patch.object(cache_mod, "CACHE_ENABLED", False):
        result = await cache_mod.get_cached(redis, "t", "Q?", ["ns"], 5)

    assert result is None
    redis.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_cached_redis_error_returns_none():
    """Redis exception → silently returns None, does not propagate."""
    from rag.cache import get_cached
    redis = AsyncMock()
    redis.get = AsyncMock(side_effect=ConnectionError("Redis down"))
    result = await get_cached(redis, "tenant-1", "Q?", ["ns"], 10)
    assert result is None


# ── set_cached ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_set_cached_stores_json():
    from rag.cache import set_cached, _cache_key
    redis = _make_redis()
    await set_cached(redis, "tenant-1", "Q?", ["ns-a"], 10, _SAMPLE_RESPONSE)

    redis.set.assert_awaited_once()
    call_args = redis.set.call_args
    stored = json.loads(call_args[0][1])
    assert stored["answer"] == "Cached answer."


@pytest.mark.asyncio
async def test_set_cached_registers_key_in_namespace_index():
    from rag.cache import set_cached
    redis = _make_redis()
    await set_cached(redis, "tenant-1", "Q?", ["ns-x"], 10, _SAMPLE_RESPONSE)
    redis.sadd.assert_awaited()
    # The sadd call should include a key containing ns-x
    sadd_args = redis.sadd.call_args[0]
    assert "ns-x" in sadd_args[0]


@pytest.mark.asyncio
async def test_set_cached_disabled_no_redis_call():
    import rag.cache as cache_mod
    redis = _make_redis()
    with patch.object(cache_mod, "CACHE_ENABLED", False):
        await cache_mod.set_cached(redis, "t", "Q?", ["ns"], 5, _SAMPLE_RESPONSE)
    redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_cached_redis_error_does_not_raise():
    from rag.cache import set_cached
    redis = AsyncMock()
    redis.set = AsyncMock(side_effect=ConnectionError("down"))
    # Should not raise
    await set_cached(redis, "t", "Q?", ["ns"], 5, _SAMPLE_RESPONSE)


# ── invalidate_namespace ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalidate_namespace_deletes_registered_keys():
    from rag.cache import invalidate_namespace
    redis = _make_redis()
    cached_key = b"rag:qcache:tenant-1:abc123"
    redis.smembers = AsyncMock(return_value={cached_key})

    count = await invalidate_namespace(redis, "tenant-1", "ns-a")

    assert count == 1
    redis.delete.assert_awaited()


@pytest.mark.asyncio
async def test_invalidate_namespace_empty_index_returns_zero():
    from rag.cache import invalidate_namespace
    redis = _make_redis()
    redis.smembers = AsyncMock(return_value=set())

    count = await invalidate_namespace(redis, "tenant-1", "ns-empty")
    assert count == 0


@pytest.mark.asyncio
async def test_invalidate_namespace_redis_error_returns_zero():
    from rag.cache import invalidate_namespace
    redis = AsyncMock()
    redis.smembers = AsyncMock(side_effect=ConnectionError("down"))

    count = await invalidate_namespace(redis, "tenant-1", "ns-a")
    assert count == 0


# ── Cache key stability ───────────────────────────────────────────────────────

def test_cache_key_stable_regardless_of_namespace_order():
    from rag.cache import _cache_key
    key_ab = _cache_key("t", "Q?", ["ns-a", "ns-b"], 10)
    key_ba = _cache_key("t", "Q?", ["ns-b", "ns-a"], 10)
    assert key_ab == key_ba


def test_cache_key_differs_by_tenant():
    from rag.cache import _cache_key
    key_t1 = _cache_key("tenant-1", "Q?", ["ns"], 10)
    key_t2 = _cache_key("tenant-2", "Q?", ["ns"], 10)
    assert key_t1 != key_t2


def test_cache_key_differs_by_question():
    from rag.cache import _cache_key
    key_q1 = _cache_key("t", "Question A", ["ns"], 10)
    key_q2 = _cache_key("t", "Question B", ["ns"], 10)
    assert key_q1 != key_q2


def test_cache_key_differs_by_top_k():
    from rag.cache import _cache_key
    key_5 = _cache_key("t", "Q?", ["ns"], 5)
    key_10 = _cache_key("t", "Q?", ["ns"], 10)
    assert key_5 != key_10
