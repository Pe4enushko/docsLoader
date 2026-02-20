from __future__ import annotations

import logging

import weaviate
from weaviate.classes.query import Filter, MetadataQuery

from .models import ChunkRecord

log = logging.getLogger(__name__)


class WeaviateQueryMixin:
    def hybrid_search_chunks(self, doc_id: str, query: str, limit: int) -> list[ChunkRecord]:
        log.debug("Using Weaviate for retrieval, version=%s", getattr(weaviate, "__version__", "unknown"))
        collection = self.client.collections.get(self.CHUNKS)
        where = Filter.by_property("doc_id").equal(doc_id)
        vector = self.embed_text(query)
        log.info("Hybrid search doc_id=%s limit=%d", doc_id, limit)
        response = collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=0.5,
            filters=where,
            limit=limit,
            return_metadata=MetadataQuery(score=True, distance=True),
        )
        out = [self._to_chunk_record(o, source="hybrid") for o in response.objects]
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
