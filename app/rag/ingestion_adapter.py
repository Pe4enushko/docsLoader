from __future__ import annotations

"""Adapter boundary for guideline ingestion storage.

The ingestion pipeline must depend on this contract instead of direct SQL
operations. This allows us to swap PostgreSQL persistence with mock/test
backends while keeping pipeline stages unchanged.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.config import get_settings
from app.schemas.knowledge import ChunkCandidate, NormalizedGuidelineDocument, RuleCandidate


@dataclass(slots=True, frozen=True)
class StoredSectionRef:
    """Minimal persisted section reference used for downstream linkage checks."""

    section_id: UUID
    section_title: str


@dataclass(slots=True)
class StoredGuidelineDocument:
    """Persisted document handle returned by ingestion adapter.

    `native_document` keeps adapter-specific object (for example SQLAlchemy row)
    and is intentionally opaque to pipeline code.
    """

    document_id: UUID
    section_refs: list[StoredSectionRef] = field(default_factory=list)
    native_document: Any | None = None


@dataclass(slots=True, frozen=True)
class StoredChunkRef:
    """Minimal persisted chunk reference."""

    chunk_id: UUID
    section_id: UUID | None = None
    order_index: int = 0


@dataclass(slots=True, frozen=True)
class StoredRuleRef:
    """Minimal persisted rule reference."""

    rule_id: UUID
    section_id: UUID | None = None


class GuidelineIngestionAdapter(ABC):
    """Contract for storing guideline ingestion artifacts."""

    @abstractmethod
    def upsert_document(self, payload: NormalizedGuidelineDocument) -> StoredGuidelineDocument:
        """Persist normalized document + section hierarchy and return handle."""
        raise NotImplementedError

    @abstractmethod
    def add_chunks(
        self,
        document: StoredGuidelineDocument,
        chunks: list[ChunkCandidate],
    ) -> list[StoredChunkRef]:
        """Persist RAG chunks for one document."""
        raise NotImplementedError

    @abstractmethod
    def add_rules(
        self,
        document: StoredGuidelineDocument,
        rules: list[RuleCandidate],
    ) -> list[StoredRuleRef]:
        """Persist extracted guideline rules for one document."""
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """Flush backend state (transactional flush or no-op for mocks)."""
        raise NotImplementedError


def create_guideline_ingestion_adapter(
    *,
    session: Session | None = None,
    adapter_name: str | None = None,
) -> GuidelineIngestionAdapter:
    """Factory for selecting ingestion adapter backend from env/config."""
    settings = get_settings()
    name = (adapter_name or settings.ingestion_rag_adapter).strip().lower()

    if name == "postgres":
        if session is None:
            raise ValueError("Postgres ingestion adapter requires active SQLAlchemy session.")
        from app.rag.postgres_ingestion_adapter import PostgresGuidelineIngestionAdapter

        return PostgresGuidelineIngestionAdapter(session=session)

    if name == "mock":
        from app.rag.mock_ingestion_adapter import MockGuidelineIngestionAdapter

        return MockGuidelineIngestionAdapter()

    raise ValueError(f"Unsupported ingestion adapter: {name}. Expected 'postgres' or 'mock'.")
