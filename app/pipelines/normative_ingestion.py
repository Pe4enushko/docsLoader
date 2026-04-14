from __future__ import annotations

"""Ingestion pipeline for normative documents.

For MVP it reuses the guideline normalizer/extractor stack and persists resulting
sections/rules into dedicated normative tables.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import UUID

from sqlalchemy.orm import Session

from app.ingestion.cleaner import ClinicalTextCleaner
from app.ingestion.rule_extractor import GuidelineRuleExtractor
from app.ingestion.section_normalizer import GuidelineSectionNormalizer
from app.ingestion.tika_client import TikaClient
from app.repositories.normative_repository import NormativeRepository
from app.utils.logging import get_pipeline_logger
from app.utils.text import stable_hash


log = get_pipeline_logger(__name__, "normative_ingestion_pipeline.log")


@dataclass(slots=True)
class NormativeIngestionResult:
    """Result summary for one ingested normative source document."""

    document_id: UUID
    rules_count: int


class NormativeIngestionPipeline:
    """Orchestrates parsing, normalization and rule persistence for normative docs."""

    def __init__(
        self,
        session: Session,
        tika_client: TikaClient | None = None,
        cleaner: ClinicalTextCleaner | None = None,
        normalizer: GuidelineSectionNormalizer | None = None,
        rule_extractor: GuidelineRuleExtractor | None = None,
    ) -> None:
        self.session = session
        self.tika_client = tika_client or TikaClient()
        self.cleaner = cleaner or ClinicalTextCleaner()
        self.normalizer = normalizer or GuidelineSectionNormalizer()
        self.rule_extractor = rule_extractor or GuidelineRuleExtractor()
        self.repo = NormativeRepository(session)

    def ingest_file(self, source_file: str | Path) -> NormativeIngestionResult:
        """Run full ingestion flow for a normative file."""
        source_path = str(source_file)
        total_started = perf_counter()
        log.info("Normative ingestion started | file=%s", source_path)

        step_started = perf_counter()
        text = self.tika_client.parse_to_text(source_path)
        log.info(
            "Normative ingestion step done | step=parse_tika | file=%s | chars=%s | elapsed_ms=%.1f",
            source_path,
            len(text),
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        clean_text = self.cleaner.clean(text)
        log.info(
            "Normative ingestion step done | step=clean_text | file=%s | chars=%s | elapsed_ms=%.1f",
            source_path,
            len(clean_text),
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        normalized = self.normalizer.normalize(source_path=source_path, extracted_text=clean_text)
        log.info(
            "Normative ingestion step done | step=normalize_sections | file=%s | sections_top=%s | elapsed_ms=%.1f",
            source_path,
            len(normalized.sections),
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        document = self.repo.create_document(
            title=normalized.title_page.title,
            source_path=source_path,
            checksum=stable_hash(f"normative:{source_path}:{clean_text[:5000]}"),
            metadata={"icd10_codes": normalized.title_page.icd10_codes},
        )
        log.info(
            "Normative ingestion step done | step=store_document | file=%s | doc_id=%s | elapsed_ms=%.1f",
            source_path,
            document.id,
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        section = self.repo.create_section(
            document_id=document.id,
            section_title="Main Text",
            raw_text=clean_text,
            cleaned_text=clean_text,
            order_index=0,
        )
        log.info(
            "Normative ingestion step done | step=store_section | file=%s | doc_id=%s | section_id=%s | elapsed_ms=%.1f",
            source_path,
            document.id,
            section.id,
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        rules = self.rule_extractor.extract(normalized)
        log.info(
            "Normative ingestion step done | step=extract_rules | file=%s | rules=%s | elapsed_ms=%.1f",
            source_path,
            len(rules),
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        rule_rows = self.repo.add_rules(document_id=document.id, section_id=section.id, rules=rules)
        log.info(
            "Normative ingestion step done | step=store_rules | file=%s | doc_id=%s | rules=%s | elapsed_ms=%.1f",
            source_path,
            document.id,
            len(rule_rows),
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        self.session.flush()
        log.info(
            "Normative ingestion step done | step=flush | file=%s | doc_id=%s | elapsed_ms=%.1f",
            source_path,
            document.id,
            (perf_counter() - step_started) * 1000,
        )

        log.info(
            "Normative ingestion completed | file=%s | doc_id=%s | rules=%s | elapsed_ms=%.1f",
            source_path,
            document.id,
            len(rule_rows),
            (perf_counter() - total_started) * 1000,
        )

        return NormativeIngestionResult(document_id=document.id, rules_count=len(rule_rows))
