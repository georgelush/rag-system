"""
rag/chunkers/custom.py — User-defined chunker with configurable separators.

The caller provides split_on — an ordered list of separator strings.
The algorithm tries each separator from highest to lowest priority and
cuts at the rightmost match within the current window.
Falls back to a hard cut only when no separator is found.

Example config:
    {
      "strategy": "custom",
      "chunk_size": 1500,
      "overlap": 150,
      "split_on": ["\n\n", "\n", ". ", " "],
      "min_chunk_size": 100
    }
"""

from rag.chunkers.base import BaseChunker
from rag.models import Chunk


class CustomChunker(BaseChunker):
    """User-controlled chunking with a priority-ordered separator list."""

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 100,
        min_chunk_size: int = 50,
        split_on: list[str] | None = None,
    ) -> None:
        super().__init__(chunk_size, overlap, min_chunk_size)
        self.split_on = split_on or ["\n\n", "\n", ". ", " "]

    def _best_split(self, text: str, start: int, end: int) -> int:
        """Find the best split point in text[start:end] using split_on priority."""
        for sep in self.split_on:
            pos = text.rfind(sep, start, end)
            if pos > start:
                return pos + len(sep)
        return end  # hard cut — no separator found in window

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

            if end < len(text):
                end = self._best_split(text, start, end)

            window = text[start:end].strip()
            if len(window) >= self.min_chunk_size:
                chunks.append(self._make_chunk(window, source_id, namespace_id, language))

            # Apply overlap — never move backward
            start = max(start + 1, end - self.overlap)

        return chunks
