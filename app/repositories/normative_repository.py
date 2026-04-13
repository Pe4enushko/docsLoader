from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.knowledge import NormativeDocument, NormativeRule, NormativeSection
from app.schemas.knowledge import RuleCandidate


class NormativeRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_document(
        self,
        title: str,
        source_path: str,
        checksum: str,
        metadata: dict,
    ) -> NormativeDocument:
        doc = NormativeDocument(
            title=title,
            source_path=source_path,
            checksum=checksum,
            metadata_json=metadata,
        )
        self.session.add(doc)
        self.session.flush()
        return doc

    def create_section(
        self,
        document_id,
        section_title: str,
        raw_text: str,
        cleaned_text: str,
        order_index: int,
        level: int = 1,
    ) -> NormativeSection:
        section = NormativeSection(
            document_id=document_id,
            section_title=section_title,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            order_index=order_index,
            level=level,
        )
        self.session.add(section)
        self.session.flush()
        return section

    def add_rules(self, document_id, rules: list[RuleCandidate], section_id=None) -> list[NormativeRule]:
        rows: list[NormativeRule] = []
        for rule in rules:
            row = NormativeRule(
                document_id=document_id,
                section_id=section_id,
                topic=rule.topic,
                rule_type=rule.rule_type,
                statement=rule.statement,
                conditions=rule.conditions,
                audit_targets=rule.audit_targets,
                source_quote=rule.source_quote,
                source_section=rule.source_section,
                metadata_json=rule.metadata,
            )
            self.session.add(row)
            rows.append(row)
        self.session.flush()
        return rows
