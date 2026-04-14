from __future__ import annotations

from sqlalchemy.orm import Session

from app.Storage import NormativeDocumentStorage, NormativeRuleStorage, NormativeSectionStorage
from app.models.knowledge import NormativeDocument, NormativeRule, NormativeSection
from app.schemas.knowledge import RuleCandidate


class NormativeRepository:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.document_storage = NormativeDocumentStorage(session)
        self.section_storage = NormativeSectionStorage(session)
        self.rule_storage = NormativeRuleStorage(session)

    def create_document(
        self,
        title: str,
        source_path: str,
        checksum: str,
        metadata: dict,
    ) -> NormativeDocument:
        return self.document_storage.create_document(
            title=title,
            source_path=source_path,
            checksum=checksum,
            metadata=metadata,
        )

    def create_section(
        self,
        document_id,
        section_title: str,
        raw_text: str,
        cleaned_text: str,
        order_index: int,
        level: int = 1,
    ) -> NormativeSection:
        return self.section_storage.create_section(
            document_id=document_id,
            section_title=section_title,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            order_index=order_index,
            level=level,
        )

    def add_rules(self, document_id, rules: list[RuleCandidate], section_id=None) -> list[NormativeRule]:
        return self.rule_storage.add_rules(document_id=document_id, rules=rules, section_id=section_id)
