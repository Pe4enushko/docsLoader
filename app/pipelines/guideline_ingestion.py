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

from sqlalchemy.orm import Session

from app.ingestion import (
    ClinicalTextCleaner,
    GuidelineChunker,
    GuidelineRuleExtractor,
    GuidelineSectionNormalizer,
    TikaClient,
)
from app.repositories.guideline_repository import GuidelineRepository
from app.services.embedding import EmbeddingProvider, OpenAIEmbeddingProvider
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
        self.embedding_provider = embedding_provider or OpenAIEmbeddingProvider()
        self.repo = GuidelineRepository(session)

    def ingest_file(self, source_file: str | Path) -> IngestionResult:
        """Run complete ingestion flow for a single guideline file."""
        source_path = str(source_file)
        log.info("Guideline ingestion started | file=%s", source_path)

        extracted_text = self.tika_client.parse_to_text(source_path)
        clean_text = self.cleaner.clean(extracted_text)
        normalized_doc = self.normalizer.normalize(source_path=source_path, extracted_text=clean_text)

        # Step 1: persist normalized document + section hierarchy.
        document = self.repo.upsert_document(normalized_doc)
        self.session.flush()

        # Step 2: build chunks and enrich them with embeddings for RAG retrieval.
        chunks = self.chunker.chunk_document(normalized_doc)
        if chunks:
            embeddings = self.embedding_provider.embed_texts([chunk.chunk_text for chunk in chunks])
            for idx, embedding in enumerate(embeddings):
                chunks[idx].embedding = embedding
        chunk_rows = self.repo.add_chunks(document, chunks)

        # Step 3: extract structured recommendation rules.
        rules = self.rule_extractor.extract(normalized_doc)
        rule_rows = self.repo.add_rules(document, rules)

        self.session.flush()
        log.info(
            "Guideline ingestion completed | doc_id=%s | sections=%s | chunks=%s | rules=%s",
            document.id,
            len(document.sections),
            len(chunk_rows),
            len(rule_rows),
        )

        return IngestionResult(
            document_id=str(document.id),
            sections_count=len(document.sections),
            chunks_count=len(chunk_rows),
            rules_count=len(rule_rows),
        )
