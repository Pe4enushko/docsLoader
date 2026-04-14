from __future__ import annotations

"""Write operations for final audit reports."""

from typing import Any

from sqlalchemy.orm import Session

from app.models.visit import AuditReport, VisitRecord


class AuditReportStorage:
    """Stores generated audit report payloads linked to visits."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_report(
        self,
        visit: VisitRecord,
        *,
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
