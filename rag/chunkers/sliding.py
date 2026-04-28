"""
rag/chunkers/sliding.py — Sliding window chunker (universal fallback).

Advances by (chunk_size - overlap) characters per step.
Always snaps to the nearest word boundary — never cuts a word in half.
"""

from rag.chunkers.base import BaseChunker
from rag.models import Chunk


class SlidingChunker(BaseChunker):
    """Classic sliding window that cuts at word boundaries."""

    def chunk(
        self,
        text: str,
        source_id: str,
        namespace_id: str,
        language: str = "en",
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Snap to nearest word boundary when not at end of text
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space

            window = text[start:end].strip()
            if len(window) >= self.min_chunk_size:
                chunks.append(self._make_chunk(window, source_id, namespace_id, language))

            # Advance with overlap — always move forward at least 1 char
            advance = max(1, (end - start) - self.overlap)
            start += advance

        return chunks
