from __future__ import annotations

from engine.models import Chunk, Document, Recommendation, Section
from engine.weaviate.storage import WeaviateGraphStore


class WeaviateIngestionAdapter:
    def __init__(self, store: WeaviateGraphStore):
        self.store = store

    def init_schema(self) -> None:
        self.store.init_schema()

    def close(self) -> None:
        self.store.close()

    def find_document_by_hash(self, doc_hash: str) -> bool:
        return self.store.find_document_by_hash(doc_hash)

    def upsert_document(self, doc: Document) -> str:
        return self.store.upsert_document(doc)

    def upsert_section(self, section: Section) -> str:
        return self.store.upsert_section(section)

    def embed_text(self, text: str) -> list[float]:
        return self.store.embed_text(text)

    def upsert_chunk(self, chunk: Chunk, embedding: list[float] | None = None) -> str:
        return self.store.upsert_chunk(chunk, embedding=embedding)

    def link_chunk_to_section(self, chunk_id: str, section_id: str) -> None:
        self.store.link_chunk_to_section(chunk_id, section_id)

    def link_chunk_to_document(self, chunk_id: str, doc_id: str) -> None:
        self.store.link_chunk_to_document(chunk_id, doc_id)

    def upsert_recommendation(self, rec: Recommendation, doc_id: str) -> str:
        return self.store.upsert_recommendation(rec, doc_id=doc_id)

    def link_recommendation_to_chunk(self, rec_id: str, chunk_id: str) -> None:
        self.store.link_recommendation_to_chunk(rec_id, chunk_id)

