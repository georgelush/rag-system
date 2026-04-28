"""
rag/chunkers/sentence.py — Sentence-boundary chunker.

Best for: news articles, narrative text, audio transcriptions.
Splits on sentence-ending punctuation (.!?) followed by whitespace.
Merges sentences up to chunk_size. Natural overlap: last sentence of
each chunk seeds the next.
"""

import re

from rag.chunkers.base import BaseChunker
from rag.models import Chunk

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker(BaseChunker):
    """Merges sentences up to chunk_size, uses last sentence as overlap seed."""

    def chunk(
        self,
        text: str,
        source_id: str,
        namespace_id: str,
        language: str = "en",
    ) -> list[Chunk]:
        sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]

        chunks: list[Chunk] = []
        buffer = ""

        for sentence in sentences:
            candidate = (buffer + " " + sentence).strip() if buffer else sentence

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer and len(buffer) >= self.min_chunk_size:
                    chunks.append(self._make_chunk(buffer, source_id, namespace_id, language))

                if len(sentence) > self.chunk_size:
                    # Very long sentence — hard-split with sliding
                    from rag.chunkers.sliding import SlidingChunker
                    sub = SlidingChunker(self.chunk_size, self.overlap, self.min_chunk_size)
                    chunks.extend(sub.chunk(sentence, source_id, namespace_id, language))
                    buffer = ""
                else:
                    buffer = sentence

        if buffer and len(buffer) >= self.min_chunk_size:
            chunks.append(self._make_chunk(buffer, source_id, namespace_id, language))

        return chunks
