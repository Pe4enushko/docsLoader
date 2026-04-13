from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge import GuidelineChunk, GuidelineDocument, GuidelineRule, GuidelineSection
from app.schemas.knowledge import ChunkCandidate, NormalizedGuidelineDocument, NormalizedSection, RuleCandidate


class GuidelineRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_checksum(self, checksum: str) -> GuidelineDocument | None:
        stmt = select(GuidelineDocument).where(GuidelineDocument.checksum == checksum)
        return self.session.scalar(stmt)

    def upsert_document(self, payload: NormalizedGuidelineDocument) -> GuidelineDocument:
        existing = self.get_by_checksum(payload.checksum)
        if existing:
            existing.title = payload.title_page.title
            existing.source_path = payload.source_path
            existing.icd10_codes = payload.title_page.icd10_codes
            existing.age_group = payload.title_page.age_group
            existing.publication_year = payload.title_page.publication_year
            existing.developer = payload.title_page.developer
            existing.metadata_json = payload.metadata
            existing.sections.clear()
            existing.chunks.clear()
            existing.rules.clear()
            self.session.flush()
            document = existing
        else:
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

        self._store_sections(document, payload.sections)
        return document

    def _store_sections(
        self,
        document: GuidelineDocument,
        sections: Iterable[NormalizedSection],
        parent: GuidelineSection | None = None,
    ) -> None:
        for section in sections:
            record = GuidelineSection(
                document_id=document.id,
                parent_section_id=parent.id if parent else None,
                section_type=section.section_type,
                section_title=section.section_title,
                level=section.level,
                order_index=section.order_index,
                page_from=section.page_from,
                page_to=section.page_to,
                raw_text=section.raw_text,
                cleaned_text=section.cleaned_text,
                metadata_json={
                    "items": [item.model_dump(mode="json") for item in section.items],
                },
            )
            self.session.add(record)
            self.session.flush()
            self._store_sections(document, section.subsections, parent=record)

    def add_chunks(self, document: GuidelineDocument, chunks: list[ChunkCandidate]) -> list[GuidelineChunk]:
        section_map = {section.section_title.lower(): section for section in document.sections}
        records: list[GuidelineChunk] = []

        for chunk in chunks:
            section_title = str(chunk.metadata.get("section_title", "")).lower()
            section = section_map.get(section_title)
            record = GuidelineChunk(
                document_id=document.id,
                section_id=section.id if section else None,
                chunk_type=chunk.chunk_type,
                chunk_text=chunk.chunk_text,
                token_count=chunk.token_count,
                order_index=chunk.order_index,
                embedding=chunk.embedding,
                metadata_json=chunk.metadata,
            )
            self.session.add(record)
            records.append(record)

        self.session.flush()
        return records

    def add_rules(self, document: GuidelineDocument, rules: list[RuleCandidate]) -> list[GuidelineRule]:
        section_map = {section.section_title.lower(): section for section in document.sections}
        records: list[GuidelineRule] = []

        for rule in rules:
            section = section_map.get((rule.source_section or "").lower())
            record = GuidelineRule(
                document_id=document.id,
                section_id=section.id if section else None,
                topic=rule.topic,
                population=rule.population,
                specialty=rule.specialty,
                rule_type=rule.rule_type,
                statement=rule.statement,
                conditions=rule.conditions,
                triggers=rule.triggers,
                audit_targets=rule.audit_targets,
                source_quote=rule.source_quote,
                source_section=rule.source_section,
                metadata_json=rule.metadata,
            )
            self.session.add(record)
            records.append(record)

        self.session.flush()
        return records
