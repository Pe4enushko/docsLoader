from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.enums import LLMCheckStage, VisitType


class RetrievalQuery(BaseModel):
    diagnosis_codes: list[str] = Field(default_factory=list)
    visit_type: VisitType
    specialty: str | None = None
    section_targets: list[str] = Field(default_factory=list)
    requested_check_type: LLMCheckStage
    max_chunks: int = 8


class RuleRef(BaseModel):
    rule_id: UUID
    statement: str
    source: str
    source_section: str | None = None


class ChunkRef(BaseModel):
    chunk_id: UUID
    text: str
    source: str
    section: str | None = None
    score: float | None = None


class RetrievalContext(BaseModel):
    selected_guideline_rules: list[RuleRef] = Field(default_factory=list)
    relevant_raw_chunks: list[ChunkRef] = Field(default_factory=list)
    short_merged_context: str
    references_metadata: list[dict[str, Any]] = Field(default_factory=list)
