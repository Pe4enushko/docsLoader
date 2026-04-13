from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.llm import StageCheckResult
from app.schemas.visit import HeuristicFlag, VisitClassificationResult


class SectionFinding(BaseModel):
    section: str
    findings: list[str] = Field(default_factory=list)


class AuditReportPayload(BaseModel):
    structural_issues: list[str] = Field(default_factory=list)
    logical_issues: list[str] = Field(default_factory=list)
    diagnostic_flags: list[str] = Field(default_factory=list)
    management_flags: list[str] = Field(default_factory=list)
    followup_adequacy: str | None = None
    human_review_required: bool = False
    section_wise_findings: list[SectionFinding] = Field(default_factory=list)
    final_summary: str
    stage_results: list[StageCheckResult] = Field(default_factory=list)
    heuristic_flags: list[HeuristicFlag] = Field(default_factory=list)
    classification: VisitClassificationResult
    references: list[dict[str, Any]] = Field(default_factory=list)
    scores: dict[str, Any] = Field(default_factory=dict)
