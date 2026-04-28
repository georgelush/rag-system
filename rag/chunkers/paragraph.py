"""
rag/chunkers/paragraph.py — Paragraph-aware chunker.

Splits on blank lines (\n\n), merges small paragraphs up to chunk_size.
Falls back to SlidingChunker for individual paragraphs that exceed chunk_size.
"""

import re

from rag.chunkers.base import BaseChunker
from rag.models import Chunk

_BLANK_LINE = re.compile(r"\n\s*\n")


class ParagraphChunker(BaseChunker):
    """Respects paragraph boundaries — never splits a paragraph unless it exceeds chunk_size."""

    def chunk(
        self,
        text: str,
        source_id: str,
        namespace_id: str,
        language: str = "en",
    ) -> list[Chunk]:
        paragraphs = [p.strip() for p in _BLANK_LINE.split(text) if p.strip()]

        chunks: list[Chunk] = []
        buffer = ""

        for para in paragraphs:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer and len(buffer) >= self.min_chunk_size:
                    chunks.append(self._make_chunk(buffer, source_id, namespace_id, language))

                if len(para) > self.chunk_size:
                    # Oversized paragraph — delegate to sliding
                    from rag.chunkers.sliding import SlidingChunker
                    sub = SlidingChunker(self.chunk_size, self.overlap, self.min_chunk_size)
                    chunks.extend(sub.chunk(para, source_id, namespace_id, language))
                    buffer = ""
                else:
                    buffer = para

        if buffer and len(buffer) >= self.min_chunk_size:
            chunks.append(self._make_chunk(buffer, source_id, namespace_id, language))

        return chunks
