"""
rag/chunkers/__init__.py — Chunker registry and auto-detector.

Usage:
    from rag.chunkers import get_chunker, PageChunker

    chunker = get_chunker(req.chunking, text=extracted_text)
    chunks  = chunker.chunk(text, source_id, namespace_id, language)

    # PDF page-aware path (when ExtractionResult.pages is available):
    if isinstance(chunker, PageChunker) and result.pages:
        chunks = chunker.chunk_pages(result.pages, source_id, namespace_id, language)
"""

import re

from rag.models import ChunkingConfig, ChunkingStrategy
from rag.chunkers.base import BaseChunker
from rag.chunkers.sliding import SlidingChunker
from rag.chunkers.paragraph import ParagraphChunker
from rag.chunkers.heading import HeadingChunker
from rag.chunkers.sentence import SentenceChunker
from rag.chunkers.page import PageChunker
from rag.chunkers.custom import CustomChunker

__all__ = [
    "get_chunker",
    "BaseChunker",
    "SlidingChunker",
    "ParagraphChunker",
    "HeadingChunker",
    "SentenceChunker",
    "PageChunker",
    "CustomChunker",
]

# ── Auto-detection signals ─────────────────────────────────────────────────────

_HEADING_SIGNAL = re.compile(
    r"(?m)^(?:#{1,6}\s+|(?:\d+\.){1,4}\s+|§\s*\d+|(?:Chapter|Section|Part)\s+\w+)",
    re.IGNORECASE,
)
_BLANK_LINE_SIGNAL = re.compile(r"\n\s*\n")


def _auto_detect(text: str) -> ChunkingStrategy:
    """Infer the best strategy from text structure (uses first 6 000 chars)."""
    sample = text[:6000]

    if len(_HEADING_SIGNAL.findall(sample)) >= 2:
        return ChunkingStrategy.heading

    if len(_BLANK_LINE_SIGNAL.findall(sample)) >= 3:
        return ChunkingStrategy.paragraph

    # Sentence heuristic: ~1 sentence per 150 chars
    if sample and (sample.count(". ") / max(len(sample), 1)) > (1 / 150):
        return ChunkingStrategy.sentence

    return ChunkingStrategy.sliding


# ── Registry ───────────────────────────────────────────────────────────────────

def get_chunker(config: ChunkingConfig, text: str = "") -> BaseChunker:
    """Return the right chunker instance for the given config.

    Args:
        config: ChunkingConfig from IngestRequest (defaults to auto/800/100).
        text:   Full extracted text — only used when strategy == "auto".

    Returns:
        A BaseChunker subclass ready to call .chunk() or .chunk_pages().
    """
    strategy = config.strategy
    if strategy == ChunkingStrategy.auto:
        strategy = _auto_detect(text)

    kwargs: dict = dict(
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        min_chunk_size=config.min_chunk_size,
    )

    match strategy:
        case ChunkingStrategy.heading:
            return HeadingChunker(**kwargs)
        case ChunkingStrategy.paragraph:
            return ParagraphChunker(**kwargs)
        case ChunkingStrategy.sentence:
            return SentenceChunker(**kwargs)
        case ChunkingStrategy.page:
            return PageChunker(**kwargs)
        case ChunkingStrategy.custom:
            return CustomChunker(**kwargs, split_on=config.split_on)
        case _:  # sliding — default fallback
            return SlidingChunker(**kwargs)
