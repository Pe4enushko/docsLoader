from __future__ import annotations

"""Ingestion pipeline for normative documents.

For MVP it reuses the guideline normalizer/extractor stack and persists resulting
sections/rules into dedicated normative tables.
"""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.ingestion.cleaner import ClinicalTextCleaner
from app.ingestion.rule_extractor import GuidelineRuleExtractor
from app.ingestion.section_normalizer import GuidelineSectionNormalizer
from app.ingestion.tika_client import TikaClient
from app.repositories.normative_repository import NormativeRepository
from app.utils.text import stable_hash


@dataclass(slots=True)
class NormativeIngestionResult:
    """Result summary for one ingested normative source document."""

    document_id: str
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
        text = self.tika_client.parse_to_text(source_path)
        clean_text = self.cleaner.clean(text)
        normalized = self.normalizer.normalize(source_path=source_path, extracted_text=clean_text)

        document = self.repo.create_document(
            title=normalized.title_page.title,
            source_path=source_path,
            checksum=stable_hash(f"normative:{source_path}:{clean_text[:5000]}"),
            metadata={"icd10_codes": normalized.title_page.icd10_codes},
        )
        section = self.repo.create_section(
            document_id=document.id,
            section_title="Main Text",
            raw_text=clean_text,
            cleaned_text=clean_text,
            order_index=0,
        )

        rules = self.rule_extractor.extract(normalized)
        rule_rows = self.repo.add_rules(document_id=document.id, section_id=section.id, rules=rules)
        self.session.flush()

        return NormativeIngestionResult(document_id=str(document.id), rules_count=len(rule_rows))
