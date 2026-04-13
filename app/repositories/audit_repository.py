from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models.visit import AuditReport, LLMCheckHistory, VisitRecord


class AuditRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_report(
        self,
        visit: VisitRecord,
        report_json: dict[str, Any],
        report_text: str,
        status: str,
        scores_json: dict[str, Any],
        llm_trace_metadata: dict[str, Any],
        readable_visit_card: str,
    ) -> AuditReport:
        row = AuditReport(
            visit_id=visit.id,
            report_json=report_json,
            report_text=report_text,
            status=status,
            scores_json=scores_json,
            llm_trace_metadata=llm_trace_metadata,
            readable_visit_card=readable_visit_card,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def create_llm_history(
        self,
        visit: VisitRecord,
        stage: str,
        prompt_version: str,
        model: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        latency_ms: int,
        token_usage_json: dict[str, Any],
        status: str,
    ) -> LLMCheckHistory:
        row = LLMCheckHistory(
            visit_id=visit.id,
            stage=stage,
            prompt_version=prompt_version,
            model=model,
            input_payload=input_payload,
            output_payload=output_payload,
            latency_ms=latency_ms,
            token_usage_json=token_usage_json,
            status=status,
        )
        self.session.add(row)
        self.session.flush()
        return row
