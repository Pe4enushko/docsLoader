from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.enums import VisitType


class CanonicalVisitSection(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)


class CanonicalVisit(BaseModel):
    meta: dict[str, Any] = Field(default_factory=dict)
    patient: dict[str, Any] = Field(default_factory=dict)
    subjective: dict[str, Any] = Field(default_factory=dict)
    objective: dict[str, Any] = Field(default_factory=dict)
    assessment: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    admin: dict[str, Any] = Field(default_factory=dict)


class HeuristicFlag(BaseModel):
    code: str
    severity: str = "warning"
    message: str
    field_path: str | None = None


class VisitClassificationResult(BaseModel):
    visit_type: VisitType
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    prompt_conditions: list[str] = Field(default_factory=list)


class VisitPreprocessResult(BaseModel):
    canonical_visit: CanonicalVisit
    flags: list[HeuristicFlag] = Field(default_factory=list)
    classification: VisitClassificationResult
    icd10_codes: list[str] = Field(default_factory=list)
    specialty: str | None = None
    age: int | None = None
    normalized_json: dict[str, Any] = Field(default_factory=dict)
