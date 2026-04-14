from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.Storage import AuditReportStorage, LLMCheckHistoryStorage
from app.models.visit import AuditReport, LLMCheckHistory, VisitRecord


class AuditRepository:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.audit_report_storage = AuditReportStorage(session)
        self.llm_history_storage = LLMCheckHistoryStorage(session)

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
        return self.audit_report_storage.create_report(
            visit,
            report_json=report_json,
            report_text=report_text,
            status=status,
            scores_json=scores_json,
            llm_trace_metadata=llm_trace_metadata,
            readable_visit_card=readable_visit_card,
        )

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
        return self.llm_history_storage.create_history(
            visit,
            stage=stage,
            prompt_version=prompt_version,
            model=model,
            input_payload=input_payload,
            output_payload=output_payload,
            latency_ms=latency_ms,
            token_usage_json=token_usage_json,
            status=status,
        )
