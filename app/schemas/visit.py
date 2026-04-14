from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.enums import VisitType


class CanonicalVisitSection(BaseModel):
    """Generic section wrapper for typed extensions of canonical visit schema."""

    data: dict[str, Any] = Field(default_factory=dict)


class CanonicalVisit(BaseModel):
    """Normalized structure used by deterministic checks, retrieval and prompts.

    Sections:
    - `meta`: visit identifiers/date/source attributes.
    - `patient`: demographics and baseline patient attributes.
    - `subjective`: complaints/anamnesis/dynamics from patient or relatives.
    - `objective`: exam findings, vitals, measurements, observed status.
    - `assessment`: diagnoses, ICD codes, clinical interpretation.
    - `plan`: prescriptions, investigations, referrals, follow-up tactics.
    - `admin`: service/provider metadata used for classification/routing.
    """

    meta: dict[str, Any] = Field(default_factory=dict)
    patient: dict[str, Any] = Field(default_factory=dict)
    subjective: dict[str, Any] = Field(default_factory=dict)
    objective: dict[str, Any] = Field(default_factory=dict)
    assessment: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    admin: dict[str, Any] = Field(default_factory=dict)


class HeuristicFlag(BaseModel):
    """Deterministic warning/error produced before LLM stages."""

    code: str
    severity: str = "warning"
    message: str
    field_path: str | None = None


class VisitClassificationResult(BaseModel):
    """Result of visit-type classifier used for prompt conditioning."""

    visit_type: VisitType
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    prompt_conditions: list[str] = Field(default_factory=list)


class VisitPreprocessResult(BaseModel):
    """Deterministic preprocessing bundle passed to all subsequent stages."""

    # Canonicalized visit card.
    canonical_visit: CanonicalVisit
    # Heuristic findings independent from LLM.
    flags: list[HeuristicFlag] = Field(default_factory=list)
    # Predicted visit type + prompt injection conditions.
    classification: VisitClassificationResult
    # Extracted ICD-10 codes for retrieval query.
    icd10_codes: list[str] = Field(default_factory=list)
    # Specialty hint used by retrieval adapter.
    specialty: str | None = None
    # Parsed age used in report and filtering.
    age: int | None = None
    # JSON snapshot persisted into DB (`visit_records.normalized_json`).
    normalized_json: dict[str, Any] = Field(default_factory=dict)
