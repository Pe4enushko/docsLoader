from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import weaviate
from weaviate.classes.query import Filter

from .models import Chunk, Document, Entity, Recommendation, Section
from .utils import stable_hash

log = logging.getLogger(__name__)


class WeaviateUpsertMixin:
    def find_document_by_hash(self, doc_hash: str) -> dict[str, Any] | None:
        log.debug("Using Weaviate for upsert/hash check, version=%s", getattr(weaviate, "__version__", "unknown"))
        collection = self.client.collections.get(self.DOCS)
        result = collection.query.fetch_objects(filters=Filter.by_property("hash").equal(doc_hash), limit=1)
        if not result.objects:
            return None
        return result.objects[0].properties

    def upsert_document(self, doc: Document) -> str:
        object_uuid = self._uuid(f"document:{doc.doc_id}")
        properties = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "year": doc.year,
            "specialty": doc.specialty,
            "source_url": doc.source_url,
            "hash": doc.hash,
            "created_at": doc.created_at.astimezone(timezone.utc).isoformat(),
        }
        log.info("Upsert document doc_id=%s", doc.doc_id)
        return self._put(self.DOCS, object_uuid, properties)

    def upsert_section(self, section: Section) -> str:
        section_id = section.section_id or stable_hash(f"{section.doc_id}|{section.path}")
        object_uuid = self._uuid(f"section:{section_id}")
        properties = {
            "section_id": section_id,
            "doc_id": section.doc_id,
            "path": section.path,
            "order": section.order,
            "level": section.level,
            "page_start": section.page_start,
            "page_end": section.page_end,
            "document_id": self._uuid(f"document:{section.doc_id}"),
        }
        self._put(self.SECTIONS, object_uuid, properties)
        log.info("Upsert section doc_id=%s path=%s", section.doc_id, section.path)
        return section_id

    def upsert_chunk(self, chunk: Chunk, embedding: list[float] | None = None) -> str:
        chunk_hash = chunk.chunk_hash or stable_hash(chunk.chunk_text)
        maybe_existing = self.client.collections.get(self.CHUNKS).query.fetch_objects(
            filters=Filter.all_of([
                Filter.by_property("doc_id").equal(chunk.doc_id),
                Filter.by_property("chunk_hash").equal(chunk_hash),
            ]),
            limit=1,
        )
        chunk_id = chunk.chunk_id or stable_hash(f"{chunk.doc_id}|{chunk.section_path}|{chunk.page_start}|{chunk_hash}")
        if maybe_existing.objects:
            existing = maybe_existing.objects[0]
            if existing.properties.get("chunk_id"):
                return str(existing.properties["chunk_id"])

        object_uuid = self._uuid(f"chunk:{chunk_id}")
        properties = {
            "chunk_id": chunk_id,
            "doc_id": chunk.doc_id,
            "section_path": chunk.section_path,
            "section_id": stable_hash(f"{chunk.doc_id}|{chunk.section_path}"),
            "document_id": self._uuid(f"document:{chunk.doc_id}"),
            "order": chunk.order,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_text": chunk.chunk_text,
            "chunk_type": chunk.chunk_type.value,
            "token_count": chunk.token_count,
            "chunk_hash": chunk_hash,
            "entity_mentions": chunk.entity_mentions,
        }
        self._put(self.CHUNKS, object_uuid, properties, vector=embedding)
        log.info("Upsert chunk doc_id=%s chunk_id=%s", chunk.doc_id, chunk_id)
        return chunk_id

    def link_chunk_to_section(self, chunk_id: str, section_id: str) -> None:
        chunk_uuid = self._uuid(f"chunk:{chunk_id}")
        collection = self.client.collections.get(self.CHUNKS)
        obj = collection.query.fetch_object_by_id(chunk_uuid)
        props = obj.properties if obj else {}
        props["section_id"] = section_id
        collection.data.replace(chunk_uuid, properties=props)
        log.info("Linked chunk->section chunk_id=%s section_id=%s", chunk_id, section_id)

    def link_chunk_to_document(self, chunk_id: str, doc_id: str) -> None:
        chunk_uuid = self._uuid(f"chunk:{chunk_id}")
        collection = self.client.collections.get(self.CHUNKS)
        obj = collection.query.fetch_object_by_id(chunk_uuid)
        props = obj.properties if obj else {}
        props["document_id"] = self._uuid(f"document:{doc_id}")
        props["doc_id"] = doc_id
        collection.data.replace(chunk_uuid, properties=props)
        log.info("Linked chunk->document chunk_id=%s doc_id=%s", chunk_id, doc_id)

    def upsert_entity(self, entity: Entity, doc_id: str) -> str:
        entity_id = stable_hash(f"{doc_id}|{entity.type}|{entity.name.lower()}")
        entity_uuid = self._uuid(f"entity:{entity_id}")
        self._put(
            self.ENTITIES,
            entity_uuid,
            {
                "entity_id": entity_id,
                "name": entity.name,
                "type": entity.type,
                "aliases": entity.aliases,
                "doc_id": doc_id,
                "chunk_ids": [],
            },
        )
        return entity_id

    def link_entity_to_chunk(self, entity_id: str, chunk_id: str) -> None:
        entity_uuid = self._uuid(f"entity:{entity_id}")
        collection = self.client.collections.get(self.ENTITIES)
        obj = collection.query.fetch_object_by_id(entity_uuid)
        if not obj:
            return
        props = obj.properties
        chunk_ids = set(props.get("chunk_ids") or [])
        chunk_ids.add(chunk_id)
        props["chunk_ids"] = list(chunk_ids)
        collection.data.replace(entity_uuid, properties=props)

    def upsert_recommendation(self, rec: Recommendation, doc_id: str) -> str:
        rec_id = stable_hash(f"{doc_id}|{rec.statement}")
        rec_uuid = self._uuid(f"recommendation:{rec_id}")
        self._put(
            self.RECS,
            rec_uuid,
            {
                "recommendation_id": rec_id,
                "doc_id": doc_id,
                "statement": rec.statement,
                "strength": rec.strength,
                "evidence_level": rec.evidence_level,
                "population": rec.population,
                "contraindications": rec.contraindications,
                "chunk_ids": [],
            },
        )
        return rec_id

    def link_recommendation_to_chunk(self, rec_id: str, chunk_id: str) -> None:
        rec_uuid = self._uuid(f"recommendation:{rec_id}")
        collection = self.client.collections.get(self.RECS)
        obj = collection.query.fetch_object_by_id(rec_uuid)
        if not obj:
            return
        props = obj.properties
        chunk_ids = set(props.get("chunk_ids") or [])
        chunk_ids.add(chunk_id)
        props["chunk_ids"] = list(chunk_ids)
        collection.data.replace(rec_uuid, properties=props)

    def store_verdict_evaluation(
        self,
        doc_id: str,
        verdict_text: str,
        retrieved_chunk_ids: list[str],
        llm_output: dict[str, Any],
        model_name: str,
    ) -> str:
        created_at = datetime.now(timezone.utc).isoformat()
        eval_id = stable_hash(f"{doc_id}|{verdict_text}|{created_at}")
        eval_uuid = self._uuid(f"evaluation:{eval_id}")
        self._put(
            self.EVALS,
            eval_uuid,
            {
                "evaluation_id": eval_id,
                "doc_id": doc_id,
                "verdict_text": verdict_text,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "llm_output": str(llm_output),
                "model_name": model_name,
                "created_at": created_at,
            },
        )
        log.info("Stored verdict evaluation doc_id=%s eval_id=%s", doc_id, eval_id)
        return eval_id
