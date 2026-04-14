from __future__ import annotations

from uuid import uuid4

import pytest

from app.domain.enums import SectionType
from app.rag.ingestion_adapter import StoredGuidelineDocument
from app.rag.mock_ingestion_adapter import MockGuidelineIngestionAdapter
from app.schemas.knowledge import (
    ChunkCandidate,
    NormalizedGuidelineDocument,
    NormalizedSection,
    TitlePageMetadata,
    RuleCandidate,
)


def _build_document() -> NormalizedGuidelineDocument:
    return NormalizedGuidelineDocument(
        source_path="docs/sample.pdf",
        checksum="abc123checksum",
        title_page=TitlePageMetadata(title="Тестовая КР", icd10_codes=["J06.9"]),
        sections=[
            NormalizedSection(
                section_type=SectionType.DIAGNOSTICS,
                section_title="2. Диагностика",
                level=1,
                order_index=1,
                raw_text="Осмотр и диагностика",
                cleaned_text="Осмотр и диагностика",
            )
        ],
    )


def test_mock_ingestion_adapter_accepts_valid_payload() -> None:
    adapter = MockGuidelineIngestionAdapter()
    document = adapter.upsert_document(_build_document())

    chunks = adapter.add_chunks(
        document,
        [
            ChunkCandidate(
                chunk_text="Рекомендуется оценить жалобы",
                order_index=0,
                token_count=4,
                metadata={"section_title": "2. Диагностика"},
            )
        ],
    )
    rules = adapter.add_rules(
        document,
        [
            RuleCandidate(
                topic="diagnostics",
                statement="Оценить жалобы и анамнез",
                source_section="2. Диагностика",
            )
        ],
    )

    adapter.flush()

    assert document.document_id
    assert len(document.section_refs) == 1
    assert len(chunks) == 1
    assert len(rules) == 1


def test_mock_ingestion_adapter_rejects_empty_chunk_text() -> None:
    adapter = MockGuidelineIngestionAdapter()
    document = adapter.upsert_document(_build_document())

    with pytest.raises(ValueError, match="chunks validation failed"):
        adapter.add_chunks(
            document,
            [
                ChunkCandidate(
                    chunk_text="",
                    order_index=0,
                    token_count=1,
                    metadata={"section_title": "2. Диагностика"},
                )
            ],
        )


def test_mock_ingestion_adapter_rejects_unknown_document_handle() -> None:
    adapter = MockGuidelineIngestionAdapter()

    with pytest.raises(ValueError, match="Unknown mock document handle"):
        adapter.add_rules(
            document=StoredGuidelineDocument(document_id=uuid4()),
            rules=[],
        )
