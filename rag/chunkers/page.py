"""
rag/chunkers/page.py — PDF page-aware chunker.

chunk()       — fallback when page list is unavailable (delegates to Paragraph).
chunk_pages() — primary path: one chunk per PDF page, page_number set on each Chunk.
                Oversized pages are subdivided via ParagraphChunker.
"""

from rag.chunkers.base import BaseChunker
from rag.models import Chunk


class PageChunker(BaseChunker):
    """One chunk per PDF page with page_number metadata."""

    def chunk(
        self,
        text: str,
        source_id: str,
        namespace_id: str,
        language: str = "en",
    ) -> list[Chunk]:
        """Fallback when page list is not available — delegates to ParagraphChunker."""
        from rag.chunkers.paragraph import ParagraphChunker
        return ParagraphChunker(self.chunk_size, self.overlap, self.min_chunk_size).chunk(
            text, source_id, namespace_id, language
        )

    def chunk_pages(
        self,
        pages: list[str],
        source_id: str,
        namespace_id: str,
        language: str = "en",
    ) -> list[Chunk]:
        """Chunk from a list of per-page strings extracted from a PDF."""
        chunks: list[Chunk] = []

        for page_num, page_text in enumerate(pages, start=1):
            page_text = page_text.strip()
            if not page_text or len(page_text) < self.min_chunk_size:
                continue

            if len(page_text) <= self.chunk_size:
                chunks.append(
                    self._make_chunk(page_text, source_id, namespace_id, language, page_number=page_num)
                )
            else:
                from rag.chunkers.paragraph import ParagraphChunker
                sub_chunks = ParagraphChunker(
                    self.chunk_size, self.overlap, self.min_chunk_size
                ).chunk(page_text, source_id, namespace_id, language)
                for c in sub_chunks:
                    c.page_number = page_num
                    chunks.append(c)

        return chunks
