from __future__ import annotations

"""Rule-level write operations for extracted guideline recommendations."""

from sqlalchemy.orm import Session

from app.models.knowledge import GuidelineDocument, GuidelineRule
from app.schemas.knowledge import RuleCandidate


class GuidelineRuleStorage:
    """Stores extracted rule rows and links them to source sections."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_rules(
        self,
        document: GuidelineDocument,
        rules: list[RuleCandidate],
    ) -> list[GuidelineRule]:
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
