from __future__ import annotations

from typing import Protocol

from engine.models import ChunkRecord


class KnowledgeGraphAdapter(Protocol):
    def hybrid_search_chunks(self, doc_id: str, query: str, limit: int) -> list[ChunkRecord]:
        ...

    def fetch_chunks_by_ids(self, doc_id: str, chunk_ids: list[str]) -> list[ChunkRecord]:
        ...

    def fetch_section_neighbors(self, doc_id: str, section_path: str, center_order: int, limit: int) -> list[ChunkRecord]:
        ...

    def fetch_chunks_by_entity_mentions(self, doc_id: str, entities: list[str], limit: int) -> list[ChunkRecord]:
        ...

    def fetch_chunks_supported_by_recommendations(self, doc_id: str, chunk_ids: list[str], limit: int) -> list[ChunkRecord]:
        ...

