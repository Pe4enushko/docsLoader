from __future__ import annotations

"""Document-level write operations for guideline ingestion."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge import GuidelineDocument
from app.schemas.knowledge import NormalizedGuidelineDocument


class GuidelineDocumentStorage:
    """Handles insert/update/reset operations for guideline documents."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_checksum(self, checksum: str) -> GuidelineDocument | None:
        stmt = select(GuidelineDocument).where(GuidelineDocument.checksum == checksum)
        return self.session.scalar(stmt)

    def upsert_document(self, payload: NormalizedGuidelineDocument) -> GuidelineDocument:
        existing = self.get_by_checksum(payload.checksum)
        if existing:
            return self._update_existing(existing, payload)
        return self._create_new(payload)

    def _update_existing(
        self,
        document: GuidelineDocument,
        payload: NormalizedGuidelineDocument,
    ) -> GuidelineDocument:
        document.title = payload.title_page.title
        document.source_path = payload.source_path
        document.icd10_codes = payload.title_page.icd10_codes
        document.age_group = payload.title_page.age_group
        document.publication_year = payload.title_page.publication_year
        document.developer = payload.title_page.developer
        document.metadata_json = payload.metadata

        # Existing derived artifacts are regenerated on every re-ingestion.
        document.sections.clear()
        document.chunks.clear()
        document.rules.clear()
        self.session.flush()
        return document

    def _create_new(self, payload: NormalizedGuidelineDocument) -> GuidelineDocument:
        document = GuidelineDocument(
            title=payload.title_page.title,
            source_path=payload.source_path,
            checksum=payload.checksum,
            icd10_codes=payload.title_page.icd10_codes,
            age_group=payload.title_page.age_group,
            publication_year=payload.title_page.publication_year,
            developer=payload.title_page.developer,
            metadata_json=payload.metadata,
        )
        self.session.add(document)
        self.session.flush()
        return document
