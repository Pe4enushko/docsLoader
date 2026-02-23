from __future__ import annotations

from engine.models import ChunkRecord
from engine.weaviate import WeaviateGraphStore


class WeaviateKnowledgeGraphAdapter:
    def __init__(self, store: WeaviateGraphStore):
        self.store = store

    def hybrid_search_chunks(self, doc_id: str, query: str, limit: int) -> list[ChunkRecord]:
        return self.store.hybrid_search_chunks(doc_id=doc_id, query=query, limit=limit)

    def fetch_chunks_by_ids(self, doc_id: str, chunk_ids: list[str]) -> list[ChunkRecord]:
        return self.store.fetch_chunks_by_ids(doc_id=doc_id, chunk_ids=chunk_ids)

    def fetch_section_neighbors(self, doc_id: str, section_path: str, center_order: int, limit: int) -> list[ChunkRecord]:
        return self.store.fetch_section_neighbors(
            doc_id=doc_id,
            section_path=section_path,
            center_order=center_order,
            limit=limit,
        )

    def fetch_chunks_by_entity_mentions(self, doc_id: str, entities: list[str], limit: int) -> list[ChunkRecord]:
        return self.store.fetch_chunks_by_entity_mentions(doc_id=doc_id, entities=entities, limit=limit)

    def fetch_chunks_supported_by_recommendations(self, doc_id: str, chunk_ids: list[str], limit: int) -> list[ChunkRecord]:
        return self.store.fetch_chunks_supported_by_recommendations(doc_id=doc_id, chunk_ids=chunk_ids, limit=limit)

