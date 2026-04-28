"""
tests/test_chunkers.py — Unit tests for all 7 chunking strategies.
"""

import pytest

from rag.models import ChunkingConfig, ChunkingStrategy
from rag.chunkers import (
    get_chunker,
    SlidingChunker,
    ParagraphChunker,
    HeadingChunker,
    SentenceChunker,
    PageChunker,
    CustomChunker,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def cfg(strategy: ChunkingStrategy, chunk_size=200, overlap=20, min_chunk_size=10, **kw) -> ChunkingConfig:
    return ChunkingConfig(strategy=strategy, chunk_size=chunk_size, overlap=overlap, min_chunk_size=min_chunk_size, **kw)


def chunk(strategy: ChunkingStrategy, text: str, **kw):
    c = cfg(strategy, **kw)
    return get_chunker(c, text=text).chunk(text, "src", "ns", "en")


# ── SlidingChunker ─────────────────────────────────────────────────────────────

def test_sliding_produces_multiple_chunks():
    text = "word " * 200  # 1000 chars
    chunks = chunk(ChunkingStrategy.sliding, text, chunk_size=100, overlap=10, min_chunk_size=10)
    assert len(chunks) > 1


def test_sliding_respects_word_boundaries():
    text = "hello world " * 50
    c = cfg(ChunkingStrategy.sliding, chunk_size=100, overlap=10, min_chunk_size=10)
    chunker = get_chunker(c)
    chunks = chunker.chunk(text, "src", "ns", "en")
    for ch in chunks:
        # chunks are stripped — should not end with space
        assert not ch.content.endswith(" ")


def test_sliding_min_chunk_size_filters_tiny():
    text = "short"  # 5 chars < min_chunk_size=10
    chunks = chunk(ChunkingStrategy.sliding, text, chunk_size=200, overlap=20, min_chunk_size=10)
    assert len(chunks) == 0


def test_sliding_overlap_creates_shared_content():
    # 400 chars: enough for 2+ chunks with overlap
    text = ("word " * 80).strip()
    c = cfg(ChunkingStrategy.sliding, chunk_size=200, overlap=100, min_chunk_size=50)
    chunker = get_chunker(c)
    chunks = chunker.chunk(text, "src", "ns", "en")
    assert len(chunks) >= 2
    if len(chunks) >= 2:
        words_1 = set(chunks[0].content.split())
        words_2 = set(chunks[1].content.split())
        assert len(words_1 & words_2) > 0


def test_sliding_chunk_fields_populated():
    text = "The quick brown fox jumps over the lazy dog. " * 5
    c = cfg(ChunkingStrategy.sliding, chunk_size=100, overlap=10, min_chunk_size=5)
    chunks = get_chunker(c).chunk(text, "my-source", "my-ns", "fr")
    for ch in chunks:
        assert ch.source_id == "my-source"
        assert ch.namespace_id == "my-ns"
        assert ch.metadata.get("language") == "fr"
        assert ch.chunk_id  # not empty
        assert ch.score == 0.0  # set at query time


# ── ParagraphChunker ───────────────────────────────────────────────────────────

def test_paragraph_splits_on_blank_lines():
    # Each paragraph must exceed chunk_size/2 so they don't all merge into one
    para = "word " * 30  # 150 chars per paragraph, chunk_size=100
    text = para.strip() + "\n\n" + para.strip() + "\n\n" + para.strip()
    chunks = chunk(ChunkingStrategy.paragraph, text, chunk_size=100, min_chunk_size=10)
    assert len(chunks) >= 2


def test_paragraph_merges_short_paragraphs():
    # Each paragraph is 10 chars, chunk_size=200 → should merge all into 1
    text = "\n\n".join(["Short." for _ in range(5)])
    chunks = chunk(ChunkingStrategy.paragraph, text, chunk_size=200, min_chunk_size=3)
    assert len(chunks) == 1


def test_paragraph_delegates_oversized_to_sliding():
    # One giant paragraph bigger than chunk_size → should still produce chunks
    text = "word " * 200  # ~1000 chars, no blank lines
    chunks = chunk(ChunkingStrategy.paragraph, text, chunk_size=100, overlap=10, min_chunk_size=5)
    assert len(chunks) > 1


# ── HeadingChunker ─────────────────────────────────────────────────────────────

def test_heading_detects_markdown_headings():
    text = "# Chapter One\nContent of chapter one.\n\n## Section 1.1\nSection content.\n\n# Chapter Two\nMore content."
    chunks = chunk(ChunkingStrategy.heading, text, chunk_size=500, min_chunk_size=5)
    assert len(chunks) >= 2


def test_heading_detects_numbered_sections():
    text = "1. Introduction\nIntro text.\n\n2. Methods\nMethods text.\n\n3. Results\nResults text."
    chunks = chunk(ChunkingStrategy.heading, text, chunk_size=500, min_chunk_size=5)
    assert len(chunks) >= 2


def test_heading_falls_back_when_no_headings():
    # No headings → delegates internally; still produces chunks
    text = "Just a paragraph.\n\nAnother paragraph.\n\nAnd one more."
    chunks = chunk(ChunkingStrategy.heading, text, chunk_size=500, min_chunk_size=5)
    assert len(chunks) >= 1  # fallback must not crash


# ── SentenceChunker ────────────────────────────────────────────────────────────

def test_sentence_splits_on_sentence_boundaries():
    # Sentences total ~400 chars; chunk_size=100 should produce >= 2 chunks
    sentence = "This is a longer sentence with enough words to count. "
    text = sentence * 8
    c = cfg(ChunkingStrategy.sentence, chunk_size=100, overlap=10, min_chunk_size=20)
    chunks = get_chunker(c).chunk(text, "src", "ns", "en")
    assert len(chunks) >= 2


def test_sentence_merges_short_sentences():
    text = "Hi. OK. Yes. No. Fine."  # All short → merged into one chunk
    chunks = chunk(ChunkingStrategy.sentence, text, chunk_size=200, min_chunk_size=3)
    assert len(chunks) == 1


# ── PageChunker ────────────────────────────────────────────────────────────────

def test_page_chunk_pages_assigns_page_numbers():
    pages = ["Page one content here.", "Page two content here.", "Page three here."]
    c = cfg(ChunkingStrategy.page, chunk_size=500, min_chunk_size=5)
    chunker = PageChunker(chunk_size=500, min_chunk_size=5)
    chunks = chunker.chunk_pages(pages, "src", "ns", "en")
    # Each non-empty page should produce at least one chunk with page_number set
    page_nums = [ch.page_number for ch in chunks if ch.page_number is not None]
    assert len(page_nums) > 0
    assert 1 in page_nums  # page numbering starts at 1


def test_page_chunk_text_fallback():
    # chunk() without pages falls back to paragraph
    text = "No pages.\n\nJust text.\n\nHere."
    c = cfg(ChunkingStrategy.page, chunk_size=500, min_chunk_size=5)
    chunks = get_chunker(c).chunk(text, "src", "ns", "en")
    assert len(chunks) >= 1


# ── CustomChunker ──────────────────────────────────────────────────────────────

def test_custom_splits_on_user_separator():
    # Each part is ~120 chars so they can't be merged (chunk_size=100)
    part = "content " * 15  # 120 chars
    text = part.strip() + "|" + part.strip() + "|" + part.strip()
    c = ChunkingConfig(
        strategy=ChunkingStrategy.custom,
        chunk_size=100,
        min_chunk_size=10,
        split_on=["|"],
    )
    chunks = get_chunker(c).chunk(text, "src", "ns", "en")
    assert len(chunks) >= 2  # at least 2 parts should be distinct


def test_custom_respects_separator_priority():
    # Each paragraph ~120 chars — too big to merge into chunk_size=100
    para = "content " * 15
    text = para.strip() + "\n\n" + para.strip() + "\n\n" + para.strip()
    c = ChunkingConfig(
        strategy=ChunkingStrategy.custom,
        chunk_size=100,
        min_chunk_size=10,
        split_on=["\n\n", "\n"],
    )
    chunks = get_chunker(c).chunk(text, "src", "ns", "en")
    assert len(chunks) >= 2


# ── Auto-detection ─────────────────────────────────────────────────────────────

def test_auto_selects_heading_for_heading_text():
    text = "\n".join([
        "# Introduction", "Intro content.",
        "## Background", "Background content.",
        "# Methods", "Methods content.",
    ])
    c = ChunkingConfig(strategy=ChunkingStrategy.auto, chunk_size=500, min_chunk_size=5)
    chunker = get_chunker(c, text=text)
    assert isinstance(chunker, HeadingChunker)


def test_auto_selects_paragraph_for_paragraph_text():
    text = "\n\n".join([f"Paragraph {i}. Some content here for testing." for i in range(5)])
    c = ChunkingConfig(strategy=ChunkingStrategy.auto, chunk_size=500, min_chunk_size=5)
    chunker = get_chunker(c, text=text)
    assert isinstance(chunker, (ParagraphChunker, HeadingChunker))  # both valid for paragraph-heavy


def test_auto_falls_back_to_sliding_for_plain_text():
    text = "This is just plain text without any structure whatsoever and goes on and on."
    c = ChunkingConfig(strategy=ChunkingStrategy.auto, chunk_size=500, min_chunk_size=5)
    chunker = get_chunker(c, text=text)
    # Should be SlidingChunker or SentenceChunker (both valid for plain text)
    assert isinstance(chunker, (SlidingChunker, SentenceChunker, ParagraphChunker))


def test_get_chunker_returns_correct_type_for_each_strategy():
    strategies_and_types = [
        (ChunkingStrategy.sliding,   SlidingChunker),
        (ChunkingStrategy.paragraph, ParagraphChunker),
        (ChunkingStrategy.heading,   HeadingChunker),
        (ChunkingStrategy.sentence,  SentenceChunker),
        (ChunkingStrategy.page,      PageChunker),
    ]
    for strategy, expected_type in strategies_and_types:
        c = ChunkingConfig(strategy=strategy)
        chunker = get_chunker(c)
        assert isinstance(chunker, expected_type), f"{strategy} → expected {expected_type.__name__}"
