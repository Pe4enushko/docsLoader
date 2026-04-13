from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.enums import ChunkType, RuleType, SectionItemType, SectionType


class TitlePageMetadata(BaseModel):
    title: str
    icd10_codes: list[str] = Field(default_factory=list)
    age_group: str | None = None
    publication_year: int | None = None
    developer: str | None = None


class NormalizedSectionItem(BaseModel):
    item_type: SectionItemType
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedSection(BaseModel):
    section_type: SectionType
    section_title: str
    level: int = 1
    order_index: int = 0
    page_from: int | None = None
    page_to: int | None = None
    raw_text: str
    cleaned_text: str
    items: list[NormalizedSectionItem] = Field(default_factory=list)
    subsections: list["NormalizedSection"] = Field(default_factory=list)


class NormalizedGuidelineDocument(BaseModel):
    source_path: str
    checksum: str
    title_page: TitlePageMetadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: list[NormalizedSection] = Field(default_factory=list)


class RuleCandidate(BaseModel):
    topic: str
    population: str | None = None
    specialty: str | None = None
    rule_type: RuleType = RuleType.OTHER
    statement: str
    conditions: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    audit_targets: list[str] = Field(default_factory=list)
    source_quote: str | None = None
    source_section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkCandidate(BaseModel):
    chunk_text: str
    chunk_type: ChunkType = ChunkType.NARRATIVE
    order_index: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    section_id: UUID | None = None


NormalizedSection.model_rebuild()
