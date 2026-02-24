from __future__ import annotations

import logging
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from engine.postgres.queries import (
    DELETE_ALL_GRAPH_SQL,
    DELETE_DOCUMENT_EDGES_SQL,
    DELETE_DOCUMENT_META_NODE_SQL,
    DELETE_DOCUMENT_NODES_SQL,
    DELETE_NAMESPACE_GRAPH_SQL,
    FETCH_CHUNKS_BY_ENTITY_MENTIONS_SQL,
    FETCH_CHUNKS_BY_IDS_SQL,
    FETCH_CHUNKS_SUPPORTED_BY_RECOMMENDATIONS_SQL,
    FETCH_SECTION_NEIGHBORS_SQL,
    FIND_DOCUMENT_BY_HASH_SQL,
    HYBRID_SEARCH_CHUNKS_SQL,
    UPSERT_EDGE_SQL,
    UPSERT_NODE_SQL,
)

log = logging.getLogger(__name__)


class PostgresGraphStore:
    DOC_NAMESPACE = "__documents__"

    def __init__(self, dsn: str, embedding_dim: int):
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False

    def close(self) -> None:
        self.conn.close()

    def _vector_literal(self, values: list[float]) -> str:
        return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
        self.conn.commit()

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def init_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS age")
            except Exception:
                log.warning("AGE extension is not available, continuing with pgvector schema")
                self.conn.rollback()
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    namespace VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR({self.embedding_dim}),
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    namespace VARCHAR(255) NOT NULL,
                    source_node_id UUID REFERENCES graph_nodes(id) ON DELETE CASCADE,
                    target_node_id UUID REFERENCES graph_nodes(id) ON DELETE CASCADE,
                    relation TEXT NOT NULL,
                    weight FLOAT DEFAULT 1.0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(namespace, source_node_id, target_node_id, relation)
                )
                """
            )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_nodes_namespace_content ON graph_nodes (namespace, content)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_nodes_embedding ON graph_nodes USING hnsw (embedding vector_cosine_ops)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges (source_node_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges (target_node_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_namespace ON graph_edges (namespace)")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_doc_kind_chunk_id
                ON graph_nodes (namespace, (metadata->>'kind'), (metadata->>'chunk_id'))
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_doc_kind_section_order
                ON graph_nodes (namespace, (metadata->>'kind'), (metadata->>'section_path'), ((metadata->>'order')::int))
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_edges_namespace_relation ON graph_edges (namespace, relation)"
            )

            cur.execute(
                """
                SELECT atttypmod
                FROM pg_attribute
                WHERE attrelid = 'graph_nodes'::regclass
                  AND attname = 'embedding'
                """
            )
            row = cur.fetchone()
            if row:
                existing_dim = row[0]
                if existing_dim != self.embedding_dim:
                    self.conn.rollback()
                    raise ValueError(
                        f"graph_nodes.embedding dimension is {existing_dim}, expected {self.embedding_dim}"
                    )

        self.conn.commit()

    def clear_namespace(self, namespace: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(DELETE_NAMESPACE_GRAPH_SQL, (namespace, namespace))
        self.conn.commit()

    def clear_all(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(DELETE_ALL_GRAPH_SQL)
        self.conn.commit()

    def delete_document(self, doc_id: str) -> dict[str, int]:
        with self.conn.cursor() as cur:
            cur.execute(DELETE_DOCUMENT_EDGES_SQL, (doc_id,))
            edges_deleted = cur.rowcount if cur.rowcount is not None else 0
            cur.execute(DELETE_DOCUMENT_NODES_SQL, (doc_id,))
            nodes_deleted = cur.rowcount if cur.rowcount is not None else 0
            cur.execute(DELETE_DOCUMENT_META_NODE_SQL, (self.DOC_NAMESPACE, doc_id))
            meta_deleted = cur.rowcount if cur.rowcount is not None else 0
        self.conn.commit()
        return {
            "edges_deleted": int(edges_deleted),
            "nodes_deleted": int(nodes_deleted),
            "meta_nodes_deleted": int(meta_deleted),
        }

    def find_document_by_hash(self, doc_hash: str) -> bool:
        row = self._fetchone(FIND_DOCUMENT_BY_HASH_SQL, (self.DOC_NAMESPACE, doc_hash))
        return bool(row)

    def upsert_node(
        self,
        namespace: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                UPSERT_NODE_SQL,
                (namespace, content, self._vector_literal(embedding), Json(metadata or {})),
            )
            row = cur.fetchone()
        self.conn.commit()
        return str(row["id"])

    def upsert_edge(
        self,
        namespace: str,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                UPSERT_EDGE_SQL,
                (namespace, source_id, target_id, relation, float(weight), Json(metadata or {})),
            )
        self.conn.commit()

    def hybrid_search_chunks(self, doc_id: str, vector: list[float], limit: int) -> list[dict[str, Any]]:
        return self._fetchall(HYBRID_SEARCH_CHUNKS_SQL, (self._vector_literal(vector), doc_id, limit))

    def fetch_chunks_by_ids(self, doc_id: str, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        return self._fetchall(FETCH_CHUNKS_BY_IDS_SQL, (doc_id, chunk_ids))

    def fetch_section_neighbors(self, doc_id: str, section_path: str, center_order: int, limit: int) -> list[dict[str, Any]]:
        return self._fetchall(FETCH_SECTION_NEIGHBORS_SQL, (doc_id, section_path, center_order, limit))

    def fetch_chunks_by_entity_mentions(self, doc_id: str, entities: list[str], limit: int) -> list[dict[str, Any]]:
        if not entities:
            return []
        return self._fetchall(FETCH_CHUNKS_BY_ENTITY_MENTIONS_SQL, (doc_id, entities, limit))

    def fetch_chunks_supported_by_recommendations(self, doc_id: str, chunk_ids: list[str], limit: int) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        return self._fetchall(FETCH_CHUNKS_SUPPORTED_BY_RECOMMENDATIONS_SQL, (doc_id, chunk_ids, doc_id, doc_id, limit))
