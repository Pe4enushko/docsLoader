from __future__ import annotations

import logging
from typing import Any

import weaviate
from weaviate.classes.query import Filter, MetadataQuery

from .models import ChunkRecord

log = logging.getLogger(__name__)


class WeaviateQueryMixin:
    def search_chunks(self, doc_id: str, query: str, filters: dict[str, Any] | None, top_k: int) -> list[ChunkRecord]:
        log.debug("Using Weaviate for retrieval, version=%s", getattr(weaviate, "__version__", "unknown"))
        collection = self.client.collections.get(self.CHUNKS)
        cond = [Filter.by_property("doc_id").equal(doc_id)]
        if filters:
            if filters.get("section_prefix"):
                cond.append(Filter.by_property("section_path").like(f"{filters['section_prefix']}*"))
            if filters.get("chunk_type_allowlist"):
                cond.append(Filter.by_property("chunk_type").contains_any(filters["chunk_type_allowlist"]))
            if filters.get("page_range"):
                page_start, page_end = filters["page_range"]
                cond.append(Filter.by_property("page_start").greater_or_equal(page_start))
                cond.append(Filter.by_property("page_end").less_or_equal(page_end))

        log.info("BM25 search doc_id=%s top_k=%d", doc_id, top_k)
        response = collection.query.bm25(
            query=query,
            filters=Filter.all_of(cond),
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
        )
        return [self._to_chunk_record(o, source="bm25") for o in response.objects]

    def hybrid_search_chunks(
        self,
        doc_id: str,
        query: str,
        top_k_vector: int,
        top_k_bm25: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkRecord]:
        bm25 = self.search_chunks(doc_id=doc_id, query=query, filters=filters, top_k=top_k_bm25)

        collection = self.client.collections.get(self.CHUNKS)
        cond = [Filter.by_property("doc_id").equal(doc_id)]
        if filters:
            if filters.get("section_prefix"):
                cond.append(Filter.by_property("section_path").like(f"{filters['section_prefix']}*"))
            if filters.get("chunk_type_allowlist"):
                cond.append(Filter.by_property("chunk_type").contains_any(filters["chunk_type_allowlist"]))
            if filters.get("page_range"):
                page_start, page_end = filters["page_range"]
                cond.append(Filter.by_property("page_start").greater_or_equal(page_start))
                cond.append(Filter.by_property("page_end").less_or_equal(page_end))

        vector = self.embed_text(query)
        vector_resp = collection.query.near_vector(
            near_vector=vector,
            filters=Filter.all_of(cond),
            limit=top_k_vector,
            return_metadata=MetadataQuery(distance=True),
        )
        vector_hits = [self._to_chunk_record(o, source="vector") for o in vector_resp.objects]

        merged: dict[str, ChunkRecord] = {}
        for rec in bm25 + vector_hits:
            current = merged.get(rec.chunk_id)
            if current is None:
                merged[rec.chunk_id] = rec
                continue
            current.score = max(current.score, rec.score)
            if current.source != rec.source:
                current.source = "hybrid"

        out = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        log.info("Hybrid search doc_id=%s candidates=%d", doc_id, len(out))
        return out

    def fetch_chunks_by_ids(self, doc_id: str, chunk_ids: list[str]) -> list[ChunkRecord]:
        if not chunk_ids:
            return []
        collection = self.client.collections.get(self.CHUNKS)
        res = collection.query.fetch_objects(
            filters=Filter.all_of([
                Filter.by_property("doc_id").equal(doc_id),
                Filter.by_property("chunk_id").contains_any(chunk_ids),
            ]),
            limit=len(chunk_ids) + 4,
        )
        return [self._to_chunk_record(o, source="fetch") for o in res.objects]

    def fetch_section_neighbors(self, doc_id: str, section_path: str, center_order: int, limit: int) -> list[ChunkRecord]:
        collection = self.client.collections.get(self.CHUNKS)
        res = collection.query.fetch_objects(
            filters=Filter.all_of([
                Filter.by_property("doc_id").equal(doc_id),
                Filter.by_property("section_path").equal(section_path),
            ]),
            limit=max(8, limit * 3),
        )
        records = [self._to_chunk_record(o, source="section") for o in res.objects]
        records.sort(key=lambda x: abs(x.order - center_order))
        return records[:limit]

    def fetch_chunks_by_entity_mentions(self, doc_id: str, entities: list[str], limit: int) -> list[ChunkRecord]:
        if not entities:
            return []
        collection = self.client.collections.get(self.CHUNKS)
        res = collection.query.fetch_objects(
            filters=Filter.all_of([
                Filter.by_property("doc_id").equal(doc_id),
                Filter.by_property("entity_mentions").contains_any(entities),
            ]),
            limit=limit,
        )
        return [self._to_chunk_record(o, source="entity") for o in res.objects]

    def fetch_chunks_supported_by_recommendations(self, doc_id: str, chunk_ids: list[str], limit: int) -> list[ChunkRecord]:
        if not chunk_ids:
            return []
        rec_collection = self.client.collections.get(self.RECS)
        recs = rec_collection.query.fetch_objects(
            filters=Filter.all_of([
                Filter.by_property("doc_id").equal(doc_id),
                Filter.by_property("chunk_ids").contains_any(chunk_ids),
            ]),
            limit=limit,
        )
        expanded_ids: set[str] = set()
        for obj in recs.objects:
            for c_id in obj.properties.get("chunk_ids") or []:
                expanded_ids.add(c_id)
        return self.fetch_chunks_by_ids(doc_id=doc_id, chunk_ids=list(expanded_ids)[:limit])
