from __future__ import annotations

"""PostgreSQL-based retrieval adapter.

Builds stage-aware context by selecting guideline rules and raw chunks from SQL
storage. The output shape matches `RetrievalContext` and is ready to inject
into prompt builders.
"""

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from app.domain.enums import RuleType
from app.models.knowledge import GuidelineChunk, GuidelineRule
from app.rag.retrieval_adapter import RetrievalAdapter
from app.schemas.retrieval import ChunkRef, RetrievalContext, RetrievalQuery, RuleRef


STAGE_TO_RULE_TYPES = {
    "formal_structure_check": {RuleType.DOCUMENTATION, RuleType.QUALITY},
    "diagnosis_consistency_check": {RuleType.DIAGNOSTIC},
    "management_consistency_check": {RuleType.MANAGEMENT},
    "followup_check": {RuleType.FOLLOWUP},
    "documentation_quality_check": {RuleType.DOCUMENTATION, RuleType.QUALITY},
}


class PostgresRetrievalAdapter(RetrievalAdapter):
    """Retrieve stage-focused context from PostgreSQL entities."""

    def __init__(self, session: Session) -> None:
        """Store SQLAlchemy session used for all retrieval queries."""
        self.session = session

    def retrieve_context(self, query: RetrievalQuery) -> RetrievalContext:
        """Fetch and merge all context components for one stage request."""
        guideline_rules = self._fetch_guideline_rules(query)
        chunks = self._fetch_chunks(query)

        merged_context = self._build_context(guideline_rules, chunks)
        references = [
            {"type": "guideline_rule", "id": str(item.rule_id), "source": item.source}
            for item in guideline_rules
        ]
        references.extend({"type": "chunk", "id": str(item.chunk_id), "source": item.source} for item in chunks)

        return RetrievalContext(
            selected_guideline_rules=guideline_rules,
            relevant_raw_chunks=chunks,
            short_merged_context=merged_context,
            references_metadata=references,
        )

    def _fetch_guideline_rules(self, query: RetrievalQuery) -> list[RuleRef]:
        """Fetch guideline rules filtered by diagnosis and check stage."""
        stmt: Select = select(GuidelineRule).limit(max(4, query.max_chunks))
        stmt = self._apply_rule_filters(stmt, GuidelineRule.statement, GuidelineRule.rule_type, query)
        rows = self.session.scalars(stmt).all()

        return [
            RuleRef(
                rule_id=row.id,
                statement=row.statement,
                source="guideline",
                source_section=row.source_section,
            )
            for row in rows
        ]

    def _apply_rule_filters(self, stmt: Select, statement_field, rule_type_field, query: RetrievalQuery) -> Select:
        """Apply common SQL filters for rules by diagnosis codes and stage type."""
        if query.diagnosis_codes:
            like_filters = [statement_field.ilike(f"%{code}%") for code in query.diagnosis_codes]
            stmt = stmt.where(or_(*like_filters))

        stage_types = STAGE_TO_RULE_TYPES.get(query.requested_check_type.value)
        if stage_types:
            stmt = stmt.where(rule_type_field.in_(stage_types))

        return stmt

    def _fetch_chunks(self, query: RetrievalQuery) -> list[ChunkRef]:
        """Fetch narrative guideline chunks constrained by target section types."""
        stmt = select(GuidelineChunk).limit(query.max_chunks)

        if query.section_targets:
            or_filters = [GuidelineChunk.metadata_json["section_type"].as_string().ilike(f"%{target}%") for target in query.section_targets]
            stmt = stmt.where(or_(*or_filters))

        rows = self.session.scalars(stmt).all()
        return [
            ChunkRef(
                chunk_id=row.id,
                text=row.chunk_text,
                source="guideline_chunk",
                section=row.metadata_json.get("section_title"),
                score=None,
            )
            for row in rows
        ]

    def _build_context(
        self,
        guideline_rules: list[RuleRef],
        chunks: list[ChunkRef],
    ) -> str:
        """Compose concise merged context block for prompt injection."""
        parts: list[str] = []

        if guideline_rules:
            parts.append("Guideline rules:")
            parts.extend(f"- {item.statement}" for item in guideline_rules[:4])

        if chunks:
            parts.append("Relevant excerpts:")
            parts.extend(f"- {item.text[:220]}" for item in chunks[:4])

        return "\n".join(parts).strip()
