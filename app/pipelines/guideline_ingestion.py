from __future__ import annotations

"""Ingestion pipeline for clinical guideline documents.

Pipeline responsibilities:
1. Parse source document via Apache Tika.
2. Normalize text and section hierarchy.
3. Build RAG chunks + embeddings.
4. Extract auditable rules (LLM-assisted).
5. Persist all artifacts into PostgreSQL.
"""

from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar
from uuid import UUID

from sqlalchemy.orm import Session

from app.ingestion import (
    ClinicalTextCleaner,
    GuidelineChunker,
    GuidelineRuleExtractor,
    GuidelineSectionNormalizer,
    TikaClient,
)
from app.rag.ingestion_adapter import (
    GuidelineIngestionAdapter,
    create_guideline_ingestion_adapter,
)
from app.services.embedding import EmbeddingProvider, create_embedding_provider
from app.logger import get_pipeline_logger


log = get_pipeline_logger(__name__, "guideline_ingestion_pipeline.log")
_T = TypeVar("_T")


@dataclass(slots=True)
class IngestionResult:
    """Summary of stored artifacts after one ingestion run."""

    document_id: UUID
    sections_count: int
    chunks_count: int
    rules_count: int


class GuidelineIngestionPipeline:
    """Orchestrates document ingestion from file to DB-ready structured assets."""

    def __init__(
        self,
        session: Session | None,
        tika_client: TikaClient | None = None,
        cleaner: ClinicalTextCleaner | None = None,
        normalizer: GuidelineSectionNormalizer | None = None,
        chunker: GuidelineChunker | None = None,
        rule_extractor: GuidelineRuleExtractor | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        ingestion_adapter: GuidelineIngestionAdapter | None = None,
    ) -> None:
        """Wire pipeline dependencies (can be replaced for tests/custom backends)."""
        self.session = session
        self.tika_client = tika_client or TikaClient()
        self.cleaner = cleaner or ClinicalTextCleaner()
        self.normalizer = normalizer or GuidelineSectionNormalizer()
        self.chunker = chunker or GuidelineChunker()
        self.rule_extractor = rule_extractor or GuidelineRuleExtractor()
        self.embedding_provider = embedding_provider or create_embedding_provider()
        self.ingestion_adapter = ingestion_adapter or create_guideline_ingestion_adapter(session=session)
        log.info("Guideline ingestion embedding provider initialized | provider=%s", self.embedding_provider.__class__.__name__)
        log.info("Guideline ingestion storage adapter initialized | adapter=%s", self.ingestion_adapter.__class__.__name__)

    def ingest_file(self, source_file: str | Path) -> IngestionResult:
        """Run complete ingestion flow for a single guideline file."""
        source_path = str(source_file)
        total_started = perf_counter()
        log.info("Guideline ingestion started | file=%s", source_path)

        try:
            extracted_text = self._run_stage(
                stage="parse_tika",
                source_path=source_path,
                fn=lambda: self.tika_client.parse_to_text(source_path),
                details_fn=lambda payload: {"chars": len(payload)},
            )
            clean_text = self._run_stage(
                stage="clean_text",
                source_path=source_path,
                fn=lambda: self.cleaner.clean(extracted_text),
                details_fn=lambda payload: {"chars": len(payload)},
            )
            normalized_doc = self._run_stage(
                stage="normalize_sections",
                source_path=source_path,
                fn=lambda: self.normalizer.normalize(source_path=source_path, extracted_text=clean_text),
                details_fn=lambda payload: {"sections_top": len(payload.sections)},
            )

            # Step 1: persist normalized document + section hierarchy.
            document = self._run_stage(
                stage="store_document_sections",
                source_path=source_path,
                fn=lambda: self.ingestion_adapter.upsert_document(normalized_doc),
                details_fn=lambda payload: {"doc_id": str(payload.document_id), "sections_total": len(payload.section_refs)},
            )
            self._run_stage(
                stage="flush_after_document",
                source_path=source_path,
                fn=self.ingestion_adapter.flush,
                details_fn=lambda _payload: {"doc_id": str(document.document_id)},
            )

            # Step 2: build chunks and enrich them with embeddings for RAG retrieval.
            chunks = self._run_stage(
                stage="chunking",
                source_path=source_path,
                fn=lambda: self.chunker.chunk_document(normalized_doc),
                details_fn=lambda payload: {"chunks": len(payload)},
            )

            if chunks:
                embeddings = self._run_stage(
                    stage="embeddings",
                    source_path=source_path,
                    fn=lambda: self.embedding_provider.embed_texts([chunk.chunk_text for chunk in chunks]),
                    details_fn=lambda payload: {
                        "chunks": len(payload),
                        "vector_dim": len(payload[0]) if payload else 0,
                    },
                )
                for idx, embedding in enumerate(embeddings):
                    chunks[idx].embedding = embedding
            else:
                log.info("Guideline ingestion step skipped | step=embeddings | file=%s | reason=no_chunks", source_path)

            chunk_rows = self._run_stage(
                stage="store_chunks",
                source_path=source_path,
                fn=lambda: self.ingestion_adapter.add_chunks(document, chunks),
                details_fn=lambda payload: {"doc_id": str(document.document_id), "chunks": len(payload)},
            )

            # Step 3: extract structured recommendation rules.
            rules = self._run_stage(
                stage="extract_rules",
                source_path=source_path,
                fn=lambda: self.rule_extractor.extract(normalized_doc),
                details_fn=lambda payload: {"rules": len(payload)},
            )
            rule_rows = self._run_stage(
                stage="store_rules",
                source_path=source_path,
                fn=lambda: self.ingestion_adapter.add_rules(document, rules),
                details_fn=lambda payload: {"doc_id": str(document.document_id), "rules": len(payload)},
            )

            self._run_stage(
                stage="flush",
                source_path=source_path,
                fn=self.ingestion_adapter.flush,
                details_fn=lambda _payload: {"doc_id": str(document.document_id)},
            )
        except Exception as exc:
            log.exception("Guideline ingestion failed | file=%s | error=%s", source_path, exc)
            raise

        log.info(
            "Guideline ingestion completed | file=%s | doc_id=%s | sections=%s | chunks=%s | rules=%s | elapsed_ms=%.1f",
            source_path,
            document.document_id,
            len(document.section_refs),
            len(chunk_rows),
            len(rule_rows),
            (perf_counter() - total_started) * 1000,
        )

        return IngestionResult(
            document_id=document.document_id,
            sections_count=len(document.section_refs),
            chunks_count=len(chunk_rows),
            rules_count=len(rule_rows),
        )

    def _run_stage(
        self,
        *,
        stage: str,
        source_path: str,
        fn: Callable[[], _T],
        details_fn: Callable[[_T], dict[str, Any]] | None = None,
    ) -> _T:
        """Execute ingestion stage with consistent start/done/failed logging."""
        started = perf_counter()
        log.info("Guideline ingestion step started | step=%s | file=%s", stage, source_path)
        try:
            payload = fn()
        except Exception as exc:
            elapsed_ms = (perf_counter() - started) * 1000
            log.exception(
                "Guideline ingestion step failed | step=%s | file=%s | elapsed_ms=%.1f | error=%s",
                stage,
                source_path,
                elapsed_ms,
                exc,
            )
            raise

        elapsed_ms = (perf_counter() - started) * 1000
        details = details_fn(payload) if details_fn else {}
        log.info(
            "Guideline ingestion step done | step=%s | file=%s | elapsed_ms=%.1f | details=%s",
            stage,
            source_path,
            elapsed_ms,
            details,
        )
        return payload
