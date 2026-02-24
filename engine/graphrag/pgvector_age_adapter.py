from __future__ import annotations

from typing import Any

from langchain_ollama import OllamaEmbeddings

from engine.config import Settings
from engine.models import Chunk, ChunkRecord, ChunkType, Document, Recommendation, Section
from engine.postgres import PostgresGraphStore
from engine.utils import stable_hash


class PgvectorAgeAdapter:
    DOC_NAMESPACE = "__documents__"

    def __init__(self, dsn: str, settings: Settings | None = None):
        self.dsn = dsn
        self.settings = settings or Settings()
        self.embeddings = OllamaEmbeddings(
            model=self.settings.ollama_embed_model,
            base_url=self.settings.ollama_embed_base_url,
        )
        self._embedding_dim = len(self.embed_text("embedding-dimension-probe"))
        self.store = PostgresGraphStore(dsn=dsn, embedding_dim=self._embedding_dim)
        self._doc_node_ids: dict[str, str] = {}
        self._section_node_ids: dict[str, str] = {}
        self._chunk_node_ids: dict[str, str] = {}
        self._chunk_doc_ids: dict[str, str] = {}
        self._rec_node_ids: dict[str, str] = {}

    def _safe_chunk_type(self, value: str | None) -> ChunkType:
        if not value:
            return ChunkType.OTHER
        try:
            return ChunkType(value)
        except ValueError:
            return ChunkType.OTHER

    def _record_from_metadata(self, meta: dict[str, Any], source: str, score: float = 0.0) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=str(meta.get("chunk_id", "")),
            doc_id=str(meta.get("doc_id", "")),
            section_path=str(meta.get("section_path", "document")),
            page_start=int(meta.get("page_start", 0) or 0),
            page_end=int(meta.get("page_end", 0) or 0),
            chunk_type=self._safe_chunk_type(str(meta.get("chunk_type", ChunkType.OTHER.value))),
            chunk_text=str(meta.get("chunk_text", "")),
            score=float(score),
            source=source,
            order=int(meta.get("order", 0) or 0),
        )

    def init_schema(self) -> None:
        self.store.init_schema()

    def close(self) -> None:
        self.store.close()

    # IngestionAdapter surface
    def find_document_by_hash(self, doc_hash: str) -> bool:
        return self.store.find_document_by_hash(doc_hash)

    def upsert_document(self, doc: Document) -> str:
        node_key = f"document:{doc.doc_id}"
        node_id = self.store.upsert_node(
            namespace=self.DOC_NAMESPACE,
            content=node_key,
            embedding=self.embed_text(f"{doc.doc_id} {doc.title}"),
            metadata={
                "kind": "document",
                "doc_id": doc.doc_id,
                "title": doc.title,
                "year": doc.year,
                "specialty": doc.specialty,
                "source_url": doc.source_url,
                "metadata_json": doc.metadata_json,
                "hash": doc.hash,
                "created_at": doc.created_at.isoformat(),
            },
        )
        self._doc_node_ids[doc.doc_id] = node_id
        return doc.doc_id

    def upsert_section(self, section: Section) -> str:
        section_id = section.section_id or stable_hash(f"{section.doc_id}|{section.path}")
        node_key = f"section:{section_id}"
        node_id = self.store.upsert_node(
            namespace=section.doc_id,
            content=node_key,
            embedding=self.embed_text(section.path),
            metadata={
                "kind": "section",
                "section_id": section_id,
                "doc_id": section.doc_id,
                "path": section.path,
                "order": section.order,
                "level": section.level,
                "page_start": section.page_start,
                "page_end": section.page_end,
            },
        )
        self._section_node_ids[section_id] = node_id
        doc_node_id = self._doc_node_ids.get(section.doc_id)
        if doc_node_id:
            self.store.upsert_edge(
                namespace=section.doc_id,
                source_id=node_id,
                target_id=doc_node_id,
                relation="belongs_to_document",
            )
        return section_id

    def embed_text(self, text: str):
        return self.embeddings.embed_query(text)

    def upsert_chunk(self, chunk: Chunk, embedding: list[float] | None = None):
        chunk_hash = chunk.chunk_hash or stable_hash(chunk.chunk_text)
        chunk_id = chunk.chunk_id or stable_hash(f"{chunk.doc_id}|{chunk.section_path}|{chunk.page_start}|{chunk_hash}")
        node_key = f"chunk:{chunk_id}"
        node_id = self.store.upsert_node(
            namespace=chunk.doc_id,
            content=node_key,
            embedding=embedding or self.embed_text(chunk.chunk_text),
            metadata={
                "kind": "chunk",
                "chunk_id": chunk_id,
                "doc_id": chunk.doc_id,
                "section_path": chunk.section_path,
                "order": chunk.order,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_text": chunk.chunk_text,
                "chunk_type": chunk.chunk_type.value,
                "token_count": chunk.token_count,
                "chunk_hash": chunk_hash,
                "entity_mentions": chunk.entity_mentions,
            },
        )
        self._chunk_node_ids[chunk_id] = node_id
        self._chunk_doc_ids[chunk_id] = chunk.doc_id
        return chunk_id

    def link_chunk_to_section(self, chunk_id: str, section_id: str) -> None:
        chunk_node_id = self._chunk_node_ids.get(chunk_id)
        section_node_id = self._section_node_ids.get(section_id)
        doc_id = self._chunk_doc_ids.get(chunk_id)
        if not chunk_node_id or not section_node_id or not doc_id:
            return
        self.store.upsert_edge(
            namespace=doc_id,
            source_id=chunk_node_id,
            target_id=section_node_id,
            relation="belongs_to_section",
        )

    def link_chunk_to_document(self, chunk_id: str, doc_id: str) -> None:
        chunk_node_id = self._chunk_node_ids.get(chunk_id)
        doc_node_id = self._doc_node_ids.get(doc_id)
        if not chunk_node_id or not doc_node_id:
            return
        self.store.upsert_edge(
            namespace=doc_id,
            source_id=chunk_node_id,
            target_id=doc_node_id,
            relation="belongs_to_document",
        )

    def upsert_recommendation(self, rec: Recommendation, doc_id: str):
        rec_id = stable_hash(f"{doc_id}|{rec.statement}")
        node_key = f"recommendation:{rec_id}"
        node_id = self.store.upsert_node(
            namespace=doc_id,
            content=node_key,
            embedding=self.embed_text(rec.statement),
            metadata={
                "kind": "recommendation",
                "recommendation_id": rec_id,
                "doc_id": doc_id,
                "statement": rec.statement,
                "strength": rec.strength,
                "evidence_level": rec.evidence_level,
                "population": rec.population,
                "contraindications": rec.contraindications,
            },
        )
        self._rec_node_ids[rec_id] = node_id
        return rec_id

    def link_recommendation_to_chunk(self, rec_id: str, chunk_id: str) -> None:
        rec_node_id = self._rec_node_ids.get(rec_id)
        chunk_node_id = self._chunk_node_ids.get(chunk_id)
        doc_id = self._chunk_doc_ids.get(chunk_id)
        if not rec_node_id or not chunk_node_id or not doc_id:
            return
        self.store.upsert_edge(
            namespace=doc_id,
            source_id=rec_node_id,
            target_id=chunk_node_id,
            relation="supports_chunk",
        )

    # KnowledgeGraphAdapter surface
    def hybrid_search_chunks(self, doc_id: str, query: str, limit: int):
        vector = self.embed_text(query)
        rows = self.store.hybrid_search_chunks(doc_id=doc_id, vector=vector, limit=limit)
        return [self._record_from_metadata(r["metadata"], source="hybrid", score=float(r.get("score", 0.0) or 0.0)) for r in rows]

    def fetch_chunks_by_ids(self, doc_id: str, chunk_ids: list[str]):
        rows = self.store.fetch_chunks_by_ids(doc_id=doc_id, chunk_ids=chunk_ids)
        return [self._record_from_metadata(r["metadata"], source="fetch") for r in rows]

    def fetch_section_neighbors(self, doc_id: str, section_path: str, center_order: int, limit: int):
        rows = self.store.fetch_section_neighbors(
            doc_id=doc_id,
            section_path=section_path,
            center_order=center_order,
            limit=limit,
        )
        return [self._record_from_metadata(r["metadata"], source="section") for r in rows]

    def fetch_chunks_by_entity_mentions(self, doc_id: str, entities: list[str], limit: int):
        rows = self.store.fetch_chunks_by_entity_mentions(doc_id=doc_id, entities=entities, limit=limit)
        return [self._record_from_metadata(r["metadata"], source="entity") for r in rows]

    def fetch_chunks_supported_by_recommendations(self, doc_id: str, chunk_ids: list[str], limit: int):
        rows = self.store.fetch_chunks_supported_by_recommendations(doc_id=doc_id, chunk_ids=chunk_ids, limit=limit)
        return [self._record_from_metadata(r["metadata"], source="recommendation") for r in rows]
