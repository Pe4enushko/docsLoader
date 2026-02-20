from __future__ import annotations

import logging

import weaviate
from weaviate.classes.query import Filter

log = logging.getLogger(__name__)


class WeaviateDeletionMixin:
    def delete_chunks_by_doc_id(self, doc_id: str) -> int:
        log.debug("Using Weaviate for deletion, version=%s", getattr(weaviate, "__version__", "unknown"))
        collection = self.client.collections.get(self.CHUNKS)
        response = collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
        count = int(getattr(response, "successful", 0) or 0)
        log.info("Deleted chunks doc_id=%s count=%d", doc_id, count)
        return count

    def delete_by_doc_id(self, doc_id: str) -> None:
        self.client.collections.get(self.EVALS).data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
        self.client.collections.get(self.RECS).data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
        self.client.collections.get(self.ENTITIES).data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
        self.client.collections.get(self.SECTIONS).data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
        self.client.collections.get(self.CHUNKS).data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
        self.client.collections.get(self.DOCS).data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
        log.info("Deleted all objects by doc_id=%s", doc_id)
