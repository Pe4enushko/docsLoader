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
from pathlib import Path
from time import perf_counter

from sqlalchemy.orm import Session

from app.ingestion import (
    ClinicalTextCleaner,
    GuidelineChunker,
    GuidelineRuleExtractor,
    GuidelineSectionNormalizer,
    TikaClient,
)
from app.repositories.guideline_repository import GuidelineRepository
from app.services.embedding import EmbeddingProvider, create_embedding_provider
from app.utils.logging import get_logger


log = get_logger(__name__)


@dataclass(slots=True)
class IngestionResult:
    """Summary of stored artifacts after one ingestion run."""

    document_id: str
    sections_count: int
    chunks_count: int
    rules_count: int


class GuidelineIngestionPipeline:
    """Orchestrates document ingestion from file to DB-ready structured assets."""

    def __init__(
        self,
        session: Session,
        tika_client: TikaClient | None = None,
        cleaner: ClinicalTextCleaner | None = None,
        normalizer: GuidelineSectionNormalizer | None = None,
        chunker: GuidelineChunker | None = None,
        rule_extractor: GuidelineRuleExtractor | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Wire pipeline dependencies (can be replaced for tests/custom backends)."""
        self.session = session
        self.tika_client = tika_client or TikaClient()
        self.cleaner = cleaner or ClinicalTextCleaner()
        self.normalizer = normalizer or GuidelineSectionNormalizer()
        self.chunker = chunker or GuidelineChunker()
        self.rule_extractor = rule_extractor or GuidelineRuleExtractor()
        self.embedding_provider = embedding_provider or create_embedding_provider()
        self.repo = GuidelineRepository(session)
        log.info("Guideline ingestion embedding provider initialized | provider=%s", self.embedding_provider.__class__.__name__)

    def ingest_file(self, source_file: str | Path) -> IngestionResult:
        """Run complete ingestion flow for a single guideline file."""
        source_path = str(source_file)
        total_started = perf_counter()
        log.info("Guideline ingestion started | file=%s", source_path)

        step_started = perf_counter()
        extracted_text = self.tika_client.parse_to_text(source_path)
        log.info(
            "Guideline ingestion step done | step=parse_tika | file=%s | chars=%s | elapsed_ms=%.1f",
            source_path,
            len(extracted_text),
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        clean_text = self.cleaner.clean(extracted_text)
        log.info(
            "Guideline ingestion step done | step=clean_text | file=%s | chars=%s | elapsed_ms=%.1f",
            source_path,
            len(clean_text),
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        normalized_doc = self.normalizer.normalize(source_path=source_path, extracted_text=clean_text)
        log.info(
            "Guideline ingestion step done | step=normalize_sections | file=%s | sections_top=%s | elapsed_ms=%.1f",
            source_path,
            len(normalized_doc.sections),
            (perf_counter() - step_started) * 1000,
        )

        # Step 1: persist normalized document + section hierarchy.
        step_started = perf_counter()
        document = self.repo.upsert_document(normalized_doc)
        self.session.flush()
        log.info(
            "Guideline ingestion step done | step=store_document_sections | file=%s | doc_id=%s | sections_total=%s | elapsed_ms=%.1f",
            source_path,
            document.id,
            len(document.sections),
            (perf_counter() - step_started) * 1000,
        )

        # Step 2: build chunks and enrich them with embeddings for RAG retrieval.
        step_started = perf_counter()
        chunks = self.chunker.chunk_document(normalized_doc)
        log.info(
            "Guideline ingestion step done | step=chunking | file=%s | chunks=%s | elapsed_ms=%.1f",
            source_path,
            len(chunks),
            (perf_counter() - step_started) * 1000,
        )

        if chunks:
            step_started = perf_counter()
            embeddings = self.embedding_provider.embed_texts([chunk.chunk_text for chunk in chunks])
            for idx, embedding in enumerate(embeddings):
                chunks[idx].embedding = embedding
            log.info(
                "Guideline ingestion step done | step=embeddings | file=%s | chunks=%s | vector_dim=%s | elapsed_ms=%.1f",
                source_path,
                len(chunks),
                len(embeddings[0]) if embeddings else 0,
                (perf_counter() - step_started) * 1000,
            )

        step_started = perf_counter()
        chunk_rows = self.repo.add_chunks(document, chunks)
        log.info(
            "Guideline ingestion step done | step=store_chunks | file=%s | doc_id=%s | chunks=%s | elapsed_ms=%.1f",
            source_path,
            document.id,
            len(chunk_rows),
            (perf_counter() - step_started) * 1000,
        )

        # Step 3: extract structured recommendation rules.
        step_started = perf_counter()
        rules = self.rule_extractor.extract(normalized_doc)
        log.info(
            "Guideline ingestion step done | step=extract_rules | file=%s | rules=%s | elapsed_ms=%.1f",
            source_path,
            len(rules),
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        rule_rows = self.repo.add_rules(document, rules)
        log.info(
            "Guideline ingestion step done | step=store_rules | file=%s | doc_id=%s | rules=%s | elapsed_ms=%.1f",
            source_path,
            document.id,
            len(rule_rows),
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        self.session.flush()
        log.info(
            "Guideline ingestion step done | step=flush | file=%s | doc_id=%s | elapsed_ms=%.1f",
            source_path,
            document.id,
            (perf_counter() - step_started) * 1000,
        )
        log.info(
            "Guideline ingestion completed | file=%s | doc_id=%s | sections=%s | chunks=%s | rules=%s | elapsed_ms=%.1f",
            source_path,
            document.id,
            len(document.sections),
            len(chunk_rows),
            len(rule_rows),
            (perf_counter() - total_started) * 1000,
        )

        return IngestionResult(
            document_id=str(document.id),
            sections_count=len(document.sections),
            chunks_count=len(chunk_rows),
            rules_count=len(rule_rows),
        )
