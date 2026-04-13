from __future__ import annotations

from app.domain.enums import ChunkType
from app.schemas.knowledge import ChunkCandidate, NormalizedGuidelineDocument


class GuidelineChunker:
    def __init__(self, max_chars: int = 900, overlap_chars: int = 120) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_document(self, doc: NormalizedGuidelineDocument) -> list[ChunkCandidate]:
        chunks: list[ChunkCandidate] = []
        order = 0

        for section in self._walk_sections(doc.sections):
            text = section.cleaned_text or section.raw_text
            if not text:
                continue
            for chunk_text in self._split_text(text):
                chunk_type = ChunkType.RECOMMENDATION if "рекоменду" in chunk_text.lower() else ChunkType.NARRATIVE
                chunks.append(
                    ChunkCandidate(
                        chunk_text=chunk_text,
                        chunk_type=chunk_type,
                        order_index=order,
                        token_count=max(1, len(chunk_text.split())),
                        metadata={
                            "section_title": section.section_title,
                            "section_type": section.section_type.value,
                        },
                    )
                )
                order += 1

        return chunks

    def _walk_sections(self, sections):
        for section in sections:
            yield section
            yield from self._walk_sections(section.subsections)

    def _split_text(self, text: str) -> list[str]:
        text = text.strip()
        if len(text) <= self.max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.max_chars)
            payload = text[start:end].strip()
            if payload:
                chunks.append(payload)
            if end == len(text):
                break
            start = max(0, end - self.overlap_chars)
        return chunks
