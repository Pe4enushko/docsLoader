from __future__ import annotations

import uuid

from app.domain.enums import LLMCheckStage, RuleType, VisitType
from app.models.knowledge import GuidelineChunk, GuidelineRule
from app.rag.postgres_adapter import PostgresRetrievalAdapter
from app.schemas.retrieval import RetrievalQuery


class _ScalarResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class _FakeSession:
    def __init__(self, payloads):
        self._payloads = payloads

    def scalars(self, _stmt):
        return _ScalarResult(self._payloads.pop(0))


def test_retrieval_adapter_contract_returns_structured_context() -> None:
    guideline_rule = GuidelineRule(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        topic="Диагностика",
        rule_type=RuleType.DIAGNOSTIC,
        statement="При J06.9 рекомендуется физикальное обследование",
        conditions=[],
        triggers=[],
        audit_targets=[],
    )
    chunk = GuidelineChunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_text="Рекомендуется контроль состояния через 5 дней",
        token_count=8,
        order_index=0,
        metadata_json={"section_title": "Диагностика"},
    )

    session = _FakeSession([[guideline_rule], [chunk]])
    adapter = PostgresRetrievalAdapter(session)  # type: ignore[arg-type]

    query = RetrievalQuery(
        diagnosis_codes=["J06.9"],
        visit_type=VisitType.PRIMARY,
        specialty="therapy",
        requested_check_type=LLMCheckStage.DIAGNOSIS_CONSISTENCY_CHECK,
        section_targets=["diagnostics"],
        max_chunks=5,
    )
    context = adapter.retrieve_context(query)

    assert context.selected_guideline_rules
    assert context.relevant_raw_chunks
    assert "Guideline rules:" in context.short_merged_context
