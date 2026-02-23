from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_ollama import OllamaEmbeddings
from postgres_graph_rag.database import DatabaseManager

from engine.config import Settings
from engine.models import Chunk, ChunkRecord, ChunkType, Document, Recommendation, Section
from engine.utils import stable_hash

class PgvectorAgeAdapter:
    DOC_NAMESPACE = "__documents__"

    def __init__(self, dsn: str, settings: Settings | None = None):
        self.dsn = dsn
        self.settings = settings or Settings()
        self.db = DatabaseManager(connection_url=dsn)
        self.embeddings = OllamaEmbeddings(
            model=self.settings.ollama_embed_model,
            base_url=self.settings.ollama_embed_base_url,
        )
        self._loop = asyncio.new_event_loop()
        self._embedding_dim = len(self.embed_text("embedding-dimension-probe"))
        self._doc_node_ids: dict[str, str] = {}
        self._section_node_ids: dict[str, str] = {}
        self._chunk_node_ids: dict[str, str] = {}
        self._chunk_doc_ids: dict[str, str] = {}
        self._rec_node_ids: dict[str, str] = {}
        self.log = logging.getLogger(__name__)

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    async def _query_rows(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        await self.db._init_pool()
        assert self.db.pool is not None
        async with self.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                return await cur.fetchall()

    async def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        await self.db._init_pool()
        assert self.db.pool is not None
        async with self.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
            await conn.commit()

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
        self._run(self._execute("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        self._run(self._execute("CREATE EXTENSION IF NOT EXISTS vector"))
        try:
            self._run(self._execute("CREATE EXTENSION IF NOT EXISTS age"))
        except Exception:
            self.log.warning("AGE extension is not available, continuing with pgvector schema")
        self._run(self.db.setup_database(embedding_dimension=self._embedding_dim))
        self._run(
            self._execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_doc_kind_chunk_id
                ON graph_nodes (namespace, (metadata->>'kind'), (metadata->>'chunk_id'));
                """
            )
        )
        self._run(
            self._execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_doc_kind_section_order
                ON graph_nodes (namespace, (metadata->>'kind'), (metadata->>'section_path'), ((metadata->>'order')::int));
                """
            )
        )
        self._run(
            self._execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_edges_namespace_relation
                ON graph_edges (namespace, relation);
                """
            )
        )

    def close(self) -> None:
        try:
            self._run(self.db.close())
        finally:
            self._loop.close()

    # IngestionAdapter surface
    def find_document_by_hash(self, doc_hash: str) -> bool:
        rows = self._run(
            self._query_rows(
                """
                SELECT id
                FROM graph_nodes
                WHERE namespace = %s
                  AND metadata->>'kind' = 'document'
                  AND metadata->>'hash' = %s
                LIMIT 1
                """,
                (self.DOC_NAMESPACE, doc_hash),
            )
        )
        return bool(rows)

    def upsert_document(self, doc: Document) -> str:
        node_key = f"document:{doc.doc_id}"
        node_id = self._run(
            self.db.upsert_node(
                content=node_key,
                embedding=self.embed_text(f"{doc.doc_id} {doc.title}"),
                namespace=self.DOC_NAMESPACE,
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
        )
        self._doc_node_ids[doc.doc_id] = node_id
        return doc.doc_id

    def upsert_section(self, section: Section) -> str:
        section_id = section.section_id or stable_hash(f"{section.doc_id}|{section.path}")
        node_key = f"section:{section_id}"
        node_id = self._run(
            self.db.upsert_node(
                content=node_key,
                embedding=self.embed_text(section.path),
                namespace=section.doc_id,
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
        )
        self._section_node_ids[section_id] = node_id
        doc_node_id = self._doc_node_ids.get(section.doc_id)
        if doc_node_id:
            self._run(
                self.db.upsert_edge(
                    source_id=node_id,
                    target_id=doc_node_id,
                    relation="belongs_to_document",
                    namespace=section.doc_id,
                )
            )
        return section_id

    def embed_text(self, text: str):
        return self.embeddings.embed_query(text)

    def upsert_chunk(self, chunk: Chunk, embedding: list[float] | None = None):
        chunk_hash = chunk.chunk_hash or stable_hash(chunk.chunk_text)
        chunk_id = chunk.chunk_id or stable_hash(f"{chunk.doc_id}|{chunk.section_path}|{chunk.page_start}|{chunk_hash}")
        node_key = f"chunk:{chunk_id}"
        node_id = self._run(
            self.db.upsert_node(
                content=node_key,
                embedding=embedding or self.embed_text(chunk.chunk_text),
                namespace=chunk.doc_id,
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
        self._run(
            self.db.upsert_edge(
                source_id=chunk_node_id,
                target_id=section_node_id,
                relation="belongs_to_section",
                namespace=doc_id,
            )
        )

    def link_chunk_to_document(self, chunk_id: str, doc_id: str) -> None:
        chunk_node_id = self._chunk_node_ids.get(chunk_id)
        doc_node_id = self._doc_node_ids.get(doc_id)
        if not chunk_node_id or not doc_node_id:
            return
        self._run(
            self.db.upsert_edge(
                source_id=chunk_node_id,
                target_id=doc_node_id,
                relation="belongs_to_document",
                namespace=doc_id,
            )
        )

    def upsert_recommendation(self, rec: Recommendation, doc_id: str):
        rec_id = stable_hash(f"{doc_id}|{rec.statement}")
        node_key = f"recommendation:{rec_id}"
        node_id = self._run(
            self.db.upsert_node(
                content=node_key,
                embedding=self.embed_text(rec.statement),
                namespace=doc_id,
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
        )
        self._rec_node_ids[rec_id] = node_id
        return rec_id

    def link_recommendation_to_chunk(self, rec_id: str, chunk_id: str) -> None:
        rec_node_id = self._rec_node_ids.get(rec_id)
        chunk_node_id = self._chunk_node_ids.get(chunk_id)
        doc_id = self._chunk_doc_ids.get(chunk_id)
        if not rec_node_id or not chunk_node_id or not doc_id:
            return
        self._run(
            self.db.upsert_edge(
                source_id=rec_node_id,
                target_id=chunk_node_id,
                relation="supports_chunk",
                namespace=doc_id,
            )
        )

    # KnowledgeGraphAdapter surface
    def hybrid_search_chunks(self, doc_id: str, query: str, limit: int):
        vector = self.embed_text(query)
        rows = self._run(
            self._query_rows(
                """
                SELECT metadata, (1 - (embedding <=> %s::vector)) AS score
                FROM graph_nodes
                WHERE namespace = %s
                  AND metadata->>'kind' = 'chunk'
                ORDER BY score DESC
                LIMIT %s
                """,
                (vector, doc_id, limit),
            )
        )
        return [self._record_from_metadata(r["metadata"], source="hybrid", score=float(r.get("score", 0.0) or 0.0)) for r in rows]

    def fetch_chunks_by_ids(self, doc_id: str, chunk_ids: list[str]):
        if not chunk_ids:
            return []
        rows = self._run(
            self._query_rows(
                """
                SELECT metadata
                FROM graph_nodes
                WHERE namespace = %s
                  AND metadata->>'kind' = 'chunk'
                  AND metadata->>'chunk_id' = ANY(%s)
                """,
                (doc_id, chunk_ids),
            )
        )
        return [self._record_from_metadata(r["metadata"], source="fetch") for r in rows]

    def fetch_section_neighbors(self, doc_id: str, section_path: str, center_order: int, limit: int):
        rows = self._run(
            self._query_rows(
                """
                SELECT metadata
                FROM graph_nodes
                WHERE namespace = %s
                  AND metadata->>'kind' = 'chunk'
                  AND metadata->>'section_path' = %s
                ORDER BY ABS(((metadata->>'order')::int - %s))
                LIMIT %s
                """,
                (doc_id, section_path, center_order, limit),
            )
        )
        return [self._record_from_metadata(r["metadata"], source="section") for r in rows]

    def fetch_chunks_by_entity_mentions(self, doc_id: str, entities: list[str], limit: int):
        if not entities:
            return []
        rows = self._run(
            self._query_rows(
                """
                SELECT metadata
                FROM graph_nodes
                WHERE namespace = %s
                  AND metadata->>'kind' = 'chunk'
                  AND EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements_text(COALESCE(metadata->'entity_mentions', '[]'::jsonb)) AS m(value)
                    WHERE m.value = ANY(%s)
                  )
                LIMIT %s
                """,
                (doc_id, entities, limit),
            )
        )
        return [self._record_from_metadata(r["metadata"], source="entity") for r in rows]

    def fetch_chunks_supported_by_recommendations(self, doc_id: str, chunk_ids: list[str], limit: int):
        if not chunk_ids:
            return []
        rows = self._run(
            self._query_rows(
                """
                WITH seed_chunks AS (
                  SELECT id
                  FROM graph_nodes
                  WHERE namespace = %s
                    AND metadata->>'kind' = 'chunk'
                    AND metadata->>'chunk_id' = ANY(%s)
                ),
                seed_recommendations AS (
                  SELECT DISTINCT e.source_node_id AS rec_id
                  FROM graph_edges e
                  JOIN seed_chunks s ON s.id = e.target_node_id
                  WHERE e.namespace = %s
                    AND e.relation = 'supports_chunk'
                )
                SELECT DISTINCT c.metadata
                FROM graph_edges e
                JOIN seed_recommendations sr ON sr.rec_id = e.source_node_id
                JOIN graph_nodes c ON c.id = e.target_node_id
                WHERE e.namespace = %s
                  AND e.relation = 'supports_chunk'
                  AND c.metadata->>'kind' = 'chunk'
                LIMIT %s
                """,
                (doc_id, chunk_ids, doc_id, doc_id, limit),
            )
        )
        return [self._record_from_metadata(r["metadata"], source="recommendation") for r in rows]
