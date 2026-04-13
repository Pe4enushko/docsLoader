from __future__ import annotations

"""Legacy compatibility adapter for appointment audit entry point."""

from typing import Any

from app.models.db import get_db_session
from app.pipelines.visit_audit import VisitAuditPipeline
from app.rag.postgres_adapter import PostgresRetrievalAdapter
from app.services.llm_client import create_llm_client


def judge_appointment(visit_payload: dict[str, Any], external_id: str | None = None) -> dict[str, Any]:
    """Run staged audit pipeline and return minimal outcome identifiers."""
    with get_db_session() as session:
        retrieval = PostgresRetrievalAdapter(session)
        pipeline = VisitAuditPipeline(session=session, retrieval_adapter=retrieval, llm_client=create_llm_client())
        result = pipeline.process_one(raw_visit=visit_payload, external_id=external_id)
        return {
            "visit_id": result.visit_id,
            "report_id": result.report_id,
            "status": result.status,
        }
