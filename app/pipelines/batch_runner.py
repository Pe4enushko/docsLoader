from __future__ import annotations

"""Helpers for sequential processing of multiple visits."""

from typing import Any

from app.pipelines.visit_audit import VisitAuditPipeline, VisitAuditResult


class VisitBatchRunner:
    """Runs visit audit pipeline for a batch of payloads one-by-one."""

    def __init__(self, audit_pipeline: VisitAuditPipeline) -> None:
        self.audit_pipeline = audit_pipeline

    def process_batch(self, visits: list[dict[str, Any]]) -> list[VisitAuditResult]:
        """Process all visits sequentially and return per-item outcomes."""
        results: list[VisitAuditResult] = []
        for item in visits:
            external_id = str(item.get("id") or item.get("guid") or "") or None
            results.append(self.audit_pipeline.process_one(raw_visit=item, external_id=external_id))
        return results
