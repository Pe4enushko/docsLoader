from __future__ import annotations

"""Chunk-level write operations for RAG ingestion artifacts."""

from sqlalchemy.orm import Session

from app.models.knowledge import GuidelineChunk, GuidelineDocument
from app.schemas.knowledge import ChunkCandidate


class GuidelineChunkStorage:
    """Stores chunk rows linked to guideline documents/sections."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_chunks(
        self,
        document: GuidelineDocument,
        chunks: list[ChunkCandidate],
    ) -> list[GuidelineChunk]:
        section_map = {section.section_title.lower(): section for section in document.sections}
        records: list[GuidelineChunk] = []

        for chunk in chunks:
            section_title = str(chunk.metadata.get("section_title", "")).lower()
            section = section_map.get(section_title)
            record = GuidelineChunk(
                document_id=document.id,
                section_id=section.id if section else None,
                chunk_type=chunk.chunk_type,
                chunk_text=chunk.chunk_text,
                token_count=chunk.token_count,
                order_index=chunk.order_index,
                embedding=chunk.embedding,
                metadata_json=chunk.metadata,
            )
            self.session.add(record)
            records.append(record)

        self.session.flush()
        return records
