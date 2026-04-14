from __future__ import annotations

"""Write operations for per-stage LLM check history."""

from typing import Any

from sqlalchemy.orm import Session

from app.models.visit import LLMCheckHistory, VisitRecord


class LLMCheckHistoryStorage:
    """Stores execution trace for each staged LLM check."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_history(
        self,
        visit: VisitRecord,
        *,
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
