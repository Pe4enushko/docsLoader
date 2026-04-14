from __future__ import annotations

"""PostgreSQL ingestion adapter.

Wraps repository SQL operations behind `GuidelineIngestionAdapter` contract.
"""

from sqlalchemy.orm import Session

from app.models.knowledge import GuidelineDocument
from app.rag.ingestion_adapter import (
    GuidelineIngestionAdapter,
    StoredChunkRef,
    StoredGuidelineDocument,
    StoredRuleRef,
    StoredSectionRef,
)
from app.repositories.guideline_repository import GuidelineRepository
from app.schemas.knowledge import ChunkCandidate, NormalizedGuidelineDocument, RuleCandidate


class PostgresGuidelineIngestionAdapter(GuidelineIngestionAdapter):
    """SQL-backed ingestion adapter used in production flows."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = GuidelineRepository(session)

    def upsert_document(self, payload: NormalizedGuidelineDocument) -> StoredGuidelineDocument:
        orm_document = self.repo.upsert_document(payload)
        return StoredGuidelineDocument(
            document_id=orm_document.id,
            section_refs=[
                StoredSectionRef(section_id=section.id, section_title=section.section_title) for section in orm_document.sections
            ],
            native_document=orm_document,
        )

    def add_chunks(
        self,
        document: StoredGuidelineDocument,
        chunks: list[ChunkCandidate],
    ) -> list[StoredChunkRef]:
        orm_document = document.native_document
        if not isinstance(orm_document, GuidelineDocument):
            raise TypeError("Postgres ingestion adapter expects SQLAlchemy GuidelineDocument handle.")

        rows = self.repo.add_chunks(orm_document, chunks)
        return [StoredChunkRef(chunk_id=row.id, section_id=row.section_id, order_index=row.order_index) for row in rows]

    def add_rules(
        self,
        document: StoredGuidelineDocument,
        rules: list[RuleCandidate],
    ) -> list[StoredRuleRef]:
        orm_document = document.native_document
        if not isinstance(orm_document, GuidelineDocument):
            raise TypeError("Postgres ingestion adapter expects SQLAlchemy GuidelineDocument handle.")

        rows = self.repo.add_rules(orm_document, rules)
        return [StoredRuleRef(rule_id=row.id, section_id=row.section_id) for row in rows]

    def flush(self) -> None:
        self.session.flush()
