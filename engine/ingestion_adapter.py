from __future__ import annotations

from typing import Protocol

from engine.models import Chunk, Document, Recommendation, Section


class IngestionAdapter(Protocol):
    def init_schema(self) -> None:
        ...

    def close(self) -> None:
        ...

    def find_document_by_hash(self, doc_hash: str) -> bool:
        ...

    def upsert_document(self, doc: Document) -> str:
        ...

    def upsert_section(self, section: Section) -> str:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...

    def upsert_chunk(self, chunk: Chunk, embedding: list[float] | None = None) -> str:
        ...

    def link_chunk_to_section(self, chunk_id: str, section_id: str) -> None:
        ...

    def link_chunk_to_document(self, chunk_id: str, doc_id: str) -> None:
        ...

    def upsert_recommendation(self, rec: Recommendation, doc_id: str) -> str:
        ...

    def link_recommendation_to_chunk(self, rec_id: str, chunk_id: str) -> None:
        ...

