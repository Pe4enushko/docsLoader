from __future__ import annotations

from sqlalchemy.orm import Session

from app.Storage import (
    GuidelineChunkStorage,
    GuidelineDocumentStorage,
    GuidelineRuleStorage,
    GuidelineSectionStorage,
)
from app.models.knowledge import GuidelineChunk, GuidelineDocument, GuidelineRule
from app.schemas.knowledge import ChunkCandidate, NormalizedGuidelineDocument, RuleCandidate


class GuidelineRepository:
    """Facade for guideline persistence operations used by ingestion adapters.

    All low-level INSERT/UPDATE logic is delegated to entity-specific storage
    scripts in `app/Storage`.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.document_storage = GuidelineDocumentStorage(session)
        self.section_storage = GuidelineSectionStorage(session)
        self.chunk_storage = GuidelineChunkStorage(session)
        self.rule_storage = GuidelineRuleStorage(session)

    def get_by_checksum(self, checksum: str) -> GuidelineDocument | None:
        return self.document_storage.get_by_checksum(checksum)

    def upsert_document(self, payload: NormalizedGuidelineDocument) -> GuidelineDocument:
        document = self.document_storage.upsert_document(payload)
        self.section_storage.store_sections(document, payload.sections)
        return document

    def add_chunks(self, document: GuidelineDocument, chunks: list[ChunkCandidate]) -> list[GuidelineChunk]:
        return self.chunk_storage.add_chunks(document, chunks)

    def add_rules(self, document: GuidelineDocument, rules: list[RuleCandidate]) -> list[GuidelineRule]:
        return self.rule_storage.add_rules(document, rules)
