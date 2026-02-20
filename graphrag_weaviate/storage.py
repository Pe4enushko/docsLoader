from __future__ import annotations

import logging
import uuid
from typing import Any
from urllib.parse import urlparse

import weaviate
from langchain_ollama import OllamaEmbeddings
from weaviate.classes.config import Configure, DataType, Property

from .config import Settings
from .models import ChunkRecord, ChunkType
from .storage_deletion import WeaviateDeletionMixin
from .storage_queries import WeaviateQueryMixin
from .storage_upsert import WeaviateUpsertMixin

log = logging.getLogger(__name__)


class WeaviateGraphStore(WeaviateUpsertMixin, WeaviateQueryMixin, WeaviateDeletionMixin):
    DOCS = "Document"
    SECTIONS = "Section"
    CHUNKS = "Chunk"
    ENTITIES = "Entity"
    RECS = "Recommendation"
    EVALS = "VerdictEvaluation"

    def __init__(self, settings: Settings):
        self.settings = settings
        log.info("Connecting to Weaviate: %s", settings.weaviate_url)
        self.client = self._connect()
        self.embeddings = OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_embed_base_url)
        self.init_schema()

    def init_schema(self) -> None:
        log.info("Initializing Weaviate schema")
        self._ensure_schema()
        log.info("Weaviate schema is ready")

    def close(self) -> None:
        self.client.close()

    def embed_text(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def _connect(self):
        parsed = urlparse(self.settings.weaviate_url)
        # Local/self-hosted Weaviate only.
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
