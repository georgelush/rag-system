"""One-shot script: clean vector_store.py duplicates and add stats cache."""
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
vs_path = ROOT / "rag" / "vector_store.py"

content = vs_path.read_text(encoding="utf-8")

# --- remove duplicate block -------------------------------------------------
marker = "\ndef collection_name("
idx1 = content.index(marker)
idx2 = content.index(marker, idx1 + 1)
clean = content[:idx2].rstrip() + "\n"

# --- add json import --------------------------------------------------------
clean = clean.replace("import logging\n", "import json\nimport logging\n", 1)

# --- append new helpers ------------------------------------------------------
addon = '''

# ── Stats cache (Redis-backed) ────────────────────────────────────────────────

_STATS_CACHE_PREFIX = "rag:stats:"


async def get_stats_cached(
    client: AsyncQdrantClient,
    col: str,
    namespace_id: str,
    redis=None,
) -> NamespaceStats | None:
    """Like get_stats() but caches the result in Redis for STATS_CACHE_TTL_S seconds.

    Pass redis=None to skip caching (e.g. in tests).
    Cache is invalidated by invalidate_stats_cache() after every ingest.
    """
    from rag.config import STATS_CACHE_TTL_S  # local import avoids circular dep

    if redis is None or STATS_CACHE_TTL_S == 0:
        return await get_stats(client, col, namespace_id)

    cache_key = f"{_STATS_CACHE_PREFIX}{col}"
    try:
        raw = await redis.get(cache_key)
        if raw:
            return NamespaceStats(**json.loads(raw))
    except Exception:
        pass

    stats = await get_stats(client, col, namespace_id)
    if stats is not None:
        try:
            await redis.set(cache_key, stats.model_dump_json(), ex=STATS_CACHE_TTL_S)
        except Exception:
            pass
    return stats


async def invalidate_stats_cache(redis, col: str) -> None:
    """Delete the cached stats entry for a collection.

    Called from run_ingest() after every successful ingest so /stats always
    reflects the current chunk and source counts.
    """
    if redis is None:
        return
    try:
        await redis.delete(f"{_STATS_CACHE_PREFIX}{col}")
    except Exception:
        pass
'''
final = clean + addon
vs_path.write_text(final, encoding="utf-8")
print(f"Done. New length: {len(final)} chars")
