from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ChunkType(str, Enum):
    RECOMMENDATION = "recommendation"
    ALGORITHM = "algorithm"
    TABLE = "table"
    DEFINITION = "definition"
    EVIDENCE = "evidence"
    APPENDIX = "appendix"
    OTHER = "other"


class VerdictLabel(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    INSUFFICIENT_INFO = "insufficient_info"


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    year: int | None = None
    specialty: str | None = None
    source_url: str | None = None
    hash: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class Section:
    doc_id: str
    path: str
    order: int
    level: int
    page_start: int | None = None
    page_end: int | None = None
    section_id: str | None = None


@dataclass(slots=True)
class Chunk:
    doc_id: str
    section_path: str
    page_start: int
    page_end: int
    chunk_text: str
    chunk_type: ChunkType
    token_count: int | None = None
    embedding: list[float] | None = None
    chunk_hash: str | None = None
    chunk_id: str | None = None
    order: int = 0
    entity_mentions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Entity:
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Recommendation:
    statement: str
    strength: str | None = None
    evidence_level: str | None = None
    population: str | None = None
    contraindications: str | None = None


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    section_path: str
    page_start: int
    page_end: int
    chunk_type: ChunkType
    chunk_text: str
    score: float = 0.0
    source: str = ""
    order: int = 0


@dataclass(slots=True)
class JudgeResult:
    verdict: VerdictLabel
    explanation: str
    citations: list[dict[str, Any]]
    missing_info: list[str]
    recommended_action: str | None
    raw_output: dict[str, Any] = field(default_factory=dict)
