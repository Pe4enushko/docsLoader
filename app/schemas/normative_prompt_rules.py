from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.enums import VisitType


class RuleAppliesTo(BaseModel):
    """Target population filter of one normative rule."""

    visit_types: list[VisitType] = Field(default_factory=list)
    specialties: list[str] = Field(default_factory=list)
    age_group: str | None = None


class NormativePromptRule(BaseModel):
    """Normative rule injected into formal-structure stage prompt."""

    rule_id: str
    source: str
    rule_type: str
    applies_to: RuleAppliesTo
    targets: list[str] = Field(default_factory=list)
    expectation: str
    flag_code: str
    severity: str = "major"
    condition: str | None = None
