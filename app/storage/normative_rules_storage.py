from __future__ import annotations

"""Rule-level write operations for normative extraction artifacts."""

from uuid import UUID

from sqlalchemy.orm import Session

from app.models.knowledge import NormativeRule
from app.schemas.knowledge import RuleCandidate


class NormativeRuleStorage:
    """Stores normative rules extracted from source sections."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_rules(
        self,
        *,
        document_id: UUID,
        rules: list[RuleCandidate],
        section_id: UUID | None = None,
    ) -> list[NormativeRule]:
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
