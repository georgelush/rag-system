"""
tests/test_config.py — Tests for rag/config.py centralised configuration.
"""

import os
import pytest


# ── validate_startup ──────────────────────────────────────────────────────────

def test_validate_startup_raises_when_no_api_key(monkeypatch):
    """Missing RAG_API_KEY → RuntimeError immediately."""
    monkeypatch.setenv("RAG_API_KEY", "")
    # Re-import to pick up fresh env
    import importlib
    import rag.config as cfg
    importlib.reload(cfg)
    with pytest.raises(RuntimeError, match="RAG_API_KEY"):
        cfg.validate_startup()


def test_validate_startup_warns_missing_webhook_secret(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("RBAC_ENABLED", raising=False)

    import importlib
    import rag.config as cfg
    importlib.reload(cfg)

    warnings = cfg.validate_startup()
    assert any("WEBHOOK_SECRET" in w for w in warnings)


def test_validate_startup_warns_rbac_enabled_without_secret(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("RBAC_ENABLED", "true")
    monkeypatch.setenv("RBAC_SECRET", "")

    import importlib
    import rag.config as cfg
    importlib.reload(cfg)

    warnings = cfg.validate_startup()
    assert any("RBAC_SECRET" in w for w in warnings)


def test_validate_startup_warns_partial_langfuse(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    import importlib
    import rag.config as cfg
    importlib.reload(cfg)

    warnings = cfg.validate_startup()
    assert any("LANGFUSE" in w for w in warnings)


def test_validate_startup_clean_returns_empty_list(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("WEBHOOK_SECRET", "my-secret")
    monkeypatch.setenv("RBAC_ENABLED", "false")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    import importlib
    import rag.config as cfg
    importlib.reload(cfg)

    warnings = cfg.validate_startup()
    assert warnings == []


# ── Configurable values ───────────────────────────────────────────────────────

def test_embed_batch_size_default(monkeypatch):
    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    import importlib
    import rag.config as cfg
    importlib.reload(cfg)
    assert cfg.EMBED_BATCH_SIZE == 500


def test_embed_batch_size_custom(monkeypatch):
    monkeypatch.setenv("EMBED_BATCH_SIZE", "200")
    import importlib
    import rag.config as cfg
    importlib.reload(cfg)
    assert cfg.EMBED_BATCH_SIZE == 200


def test_stats_cache_ttl_default(monkeypatch):
    monkeypatch.delenv("STATS_CACHE_TTL_S", raising=False)
    import importlib
    import rag.config as cfg
    importlib.reload(cfg)
    assert cfg.STATS_CACHE_TTL_S == 60


def test_batch_query_max_default(monkeypatch):
    monkeypatch.delenv("BATCH_QUERY_MAX", raising=False)
    import importlib
    import rag.config as cfg
    importlib.reload(cfg)
    assert cfg.BATCH_QUERY_MAX == 10
