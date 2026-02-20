from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import weaviate
from langchain_ollama import OllamaEmbeddings
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery

from .config import Settings
from .models import Chunk, ChunkRecord, ChunkType, Document, Entity, Recommendation, Section
from .utils import stable_hash


class WeaviateGraphStore:
    DOCS = "Document"
    SECTIONS = "Section"
    CHUNKS = "Chunk"
    ENTITIES = "Entity"
    RECS = "Recommendation"
    EVALS = "VerdictEvaluation"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = self._connect()
        self.embeddings = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_embed_base_url,
        )
        self._ensure_schema()

    def close(self) -> None:
        self.client.close()

    def _connect(self):
        if self.settings.weaviate_api_key:
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=self.settings.weaviate_url,
                auth_credentials=AuthApiKey(self.settings.weaviate_api_key),
            )
        if self.settings.weaviate_url.startswith("http://localhost") or self.settings.weaviate_url.startswith("http://127.0.0.1"):
            return weaviate.connect_to_local()
        parsed = urlparse(self.settings.weaviate_url)
        return weaviate.connect_to_custom(
            http_host=parsed.hostname or "localhost",
            http_port=parsed.port or (443 if parsed.scheme == "https" else 80),
            http_secure=parsed.scheme == "https",
        )

    def _ensure_schema(self) -> None:
        self._ensure_collection(
            self.DOCS,
            [
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="year", data_type=DataType.INT),
                Property(name="specialty", data_type=DataType.TEXT),
                Property(name="source_url", data_type=DataType.TEXT),
                Property(name="hash", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.DATE),
            ],
        )
        self._ensure_collection(
            self.SECTIONS,
            [
                Property(name="section_id", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="path", data_type=DataType.TEXT),
                Property(name="order", data_type=DataType.INT),
                Property(name="level", data_type=DataType.INT),
                Property(name="page_start", data_type=DataType.INT),
                Property(name="page_end", data_type=DataType.INT),
                Property(name="document_id", data_type=DataType.TEXT),
            ],
        )
        self._ensure_collection(
            self.CHUNKS,
            [
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="section_path", data_type=DataType.TEXT),
                Property(name="section_id", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="order", data_type=DataType.INT),
                Property(name="page_start", data_type=DataType.INT),
                Property(name="page_end", data_type=DataType.INT),
                Property(name="chunk_text", data_type=DataType.TEXT),
                Property(name="chunk_type", data_type=DataType.TEXT),
                Property(name="token_count", data_type=DataType.INT),
                Property(name="chunk_hash", data_type=DataType.TEXT),
                Property(name="entity_mentions", data_type=DataType.TEXT_ARRAY),
            ],
        )
        self._ensure_collection(
            self.ENTITIES,
            [
                Property(name="entity_id", data_type=DataType.TEXT),
                Property(name="name", data_type=DataType.TEXT),
                Property(name="type", data_type=DataType.TEXT),
                Property(name="aliases", data_type=DataType.TEXT_ARRAY),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="chunk_ids", data_type=DataType.TEXT_ARRAY),
            ],
        )
        self._ensure_collection(
            self.RECS,
            [
                Property(name="recommendation_id", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="statement", data_type=DataType.TEXT),
                Property(name="strength", data_type=DataType.TEXT),
                Property(name="evidence_level", data_type=DataType.TEXT),
                Property(name="population", data_type=DataType.TEXT),
                Property(name="contraindications", data_type=DataType.TEXT),
                Property(name="chunk_ids", data_type=DataType.TEXT_ARRAY),
            ],
        )
        self._ensure_collection(
            self.EVALS,
            [
                Property(name="evaluation_id", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="verdict_text", data_type=DataType.TEXT),
                Property(name="retrieved_chunk_ids", data_type=DataType.TEXT_ARRAY),
                Property(name="llm_output", data_type=DataType.TEXT),
                Property(name="model_name", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.DATE),
            ],
        )

    def _ensure_collection(self, name: str, properties: list[Property]) -> None:
        if self.client.collections.exists(name):
            return
        self.client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=properties,
        )

    def _uuid(self, key: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def _put(self, collection_name: str, object_uuid: str, properties: dict[str, Any], vector: list[float] | None = None) -> str:
        collection = self.client.collections.get(collection_name)
        try:
            collection.data.insert(uuid=object_uuid, properties=properties, vector=vector)
            return object_uuid
        except Exception:
            collection.data.replace(uuid=object_uuid, properties=properties, vector=vector)
            return object_uuid

    def embed_text(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def find_document_by_hash(self, doc_hash: str) -> dict[str, Any] | None:
        collection = self.client.collections.get(self.DOCS)
        result = collection.query.fetch_objects(
            filters=Filter.by_property("hash").equal(doc_hash),
            limit=1,
        )
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
        chunk_id = chunk.chunk_id or stable_hash(
            f"{chunk.doc_id}|{chunk.section_path}|{chunk.page_start}|{chunk_hash}"
        )
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
        return chunk_id

    def link_chunk_to_section(self, chunk_id: str, section_id: str) -> None:
        chunk_uuid = self._uuid(f"chunk:{chunk_id}")
        collection = self.client.collections.get(self.CHUNKS)
        obj = collection.query.fetch_object_by_id(chunk_uuid)
        props = obj.properties if obj else {}
        props["section_id"] = section_id
        collection.data.replace(chunk_uuid, properties=props)

    def link_chunk_to_document(self, chunk_id: str, doc_id: str) -> None:
        chunk_uuid = self._uuid(f"chunk:{chunk_id}")
        collection = self.client.collections.get(self.CHUNKS)
        obj = collection.query.fetch_object_by_id(chunk_uuid)
        props = obj.properties if obj else {}
        props["document_id"] = self._uuid(f"document:{doc_id}")
        props["doc_id"] = doc_id
        collection.data.replace(chunk_uuid, properties=props)

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

    def search_chunks(self, doc_id: str, query: str, filters: dict[str, Any] | None, top_k: int) -> list[ChunkRecord]:
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
        return sorted(merged.values(), key=lambda x: x.score, reverse=True)

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
        expanded = list(expanded_ids)[:limit]
        return self.fetch_chunks_by_ids(doc_id=doc_id, chunk_ids=expanded)

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
        return eval_id

    def _to_chunk_record(self, obj: Any, source: str) -> ChunkRecord:
        props = obj.properties
        raw_score = 0.0
        if obj.metadata:
            if getattr(obj.metadata, "score", None) is not None:
                raw_score = float(obj.metadata.score)
            elif getattr(obj.metadata, "distance", None) is not None:
                raw_score = 1.0 - float(obj.metadata.distance)
        return ChunkRecord(
            chunk_id=props["chunk_id"],
            doc_id=props["doc_id"],
            section_path=props["section_path"],
            page_start=int(props["page_start"]),
            page_end=int(props["page_end"]),
            chunk_type=ChunkType(props["chunk_type"]),
            chunk_text=props["chunk_text"],
            score=raw_score,
            source=source,
            order=int(props.get("order", 0)),
        )
