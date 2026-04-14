from __future__ import annotations

"""Mock ingestion adapter for testing guideline ingestion pipeline.

This adapter mimics ingestion storage API but performs no SQL operations.
Instead, it validates payload quality, checks internal consistency and emits
detailed structured logs for every storage stage.
"""

import uuid
from dataclasses import dataclass

from app.config import get_settings
from app.logger import get_pipeline_logger
from app.rag.ingestion_adapter import (
    GuidelineIngestionAdapter,
    StoredChunkRef,
    StoredGuidelineDocument,
    StoredRuleRef,
    StoredSectionRef,
)
from app.schemas.knowledge import ChunkCandidate, NormalizedGuidelineDocument, NormalizedSection, RuleCandidate

log = get_pipeline_logger(__name__, "mock_rag_ingestion_adapter.log")


@dataclass(slots=True)
class _MockDocumentState:
    """In-memory state for one mock-ingested document."""

    checksum: str
    source_path: str
    section_by_title: dict[str, uuid.UUID]
    chunk_ids: list[uuid.UUID]
    rule_ids: list[uuid.UUID]


class MockGuidelineIngestionAdapter(GuidelineIngestionAdapter):
    """Validation-first ingestion adapter used in tests and dry-runs."""

    def __init__(self) -> None:
        settings = get_settings()
        self.expected_embedding_dimension = settings.embedding_dimension
        self._documents: dict[uuid.UUID, _MockDocumentState] = {}

    def upsert_document(self, payload: NormalizedGuidelineDocument) -> StoredGuidelineDocument:
        errors: list[str] = []
        warnings: list[str] = []

        if not payload.source_path.strip():
            errors.append("source_path is empty")
        if not payload.checksum.strip():
            errors.append("checksum is empty")
        if not payload.title_page.title.strip():
            errors.append("title_page.title is empty")

        section_refs: list[StoredSectionRef] = []
        for section in payload.sections:
            section_refs.extend(self._validate_and_collect_sections(section, parent_level=0, errors=errors, warnings=warnings))

        if not section_refs:
            warnings.append("no normalized sections were produced")

        if errors:
            self._log_validation_failure(stage="store_document_sections", errors=errors, warnings=warnings)
            raise ValueError(f"Mock ingestion document validation failed: {'; '.join(errors)}")

        document_id = uuid.uuid4()
        section_by_title = {
            ref.section_title.strip().lower(): ref.section_id
            for ref in section_refs
            if ref.section_title.strip()
        }
        self._documents[document_id] = _MockDocumentState(
            checksum=payload.checksum,
            source_path=payload.source_path,
            section_by_title=section_by_title,
            chunk_ids=[],
            rule_ids=[],
        )

        log.info(
            "Mock ingestion validation passed | stage=store_document_sections | doc_id=%s | sections=%s | warnings=%s",
            document_id,
            len(section_refs),
            warnings,
        )
        return StoredGuidelineDocument(document_id=document_id, section_refs=section_refs, native_document=None)

    def add_chunks(
        self,
        document: StoredGuidelineDocument,
        chunks: list[ChunkCandidate],
    ) -> list[StoredChunkRef]:
        state = self._require_document_state(document.document_id)
        errors: list[str] = []
        warnings: list[str] = []

        seen_order_indexes: set[int] = set()
        embedding_dims: set[int] = set()
        unmatched_sections = 0
        stored: list[StoredChunkRef] = []

        for index, chunk in enumerate(chunks):
            if not chunk.chunk_text.strip():
                errors.append(f"chunk[{index}] has empty chunk_text")
            if chunk.token_count < 0:
                errors.append(f"chunk[{index}] has negative token_count={chunk.token_count}")
            if chunk.order_index in seen_order_indexes:
                warnings.append(f"chunk[{index}] duplicates order_index={chunk.order_index}")
            seen_order_indexes.add(chunk.order_index)

            section_title = str(chunk.metadata.get("section_title", "")).strip().lower()
            linked_section_id = state.section_by_title.get(section_title)
            if section_title and linked_section_id is None:
                unmatched_sections += 1
                warnings.append(f"chunk[{index}] references unknown section_title='{section_title}'")

            if chunk.embedding is not None:
                vector_dim = len(chunk.embedding)
                embedding_dims.add(vector_dim)
                if vector_dim == 0:
                    errors.append(f"chunk[{index}] has empty embedding vector")

            chunk_id = uuid.uuid4()
            state.chunk_ids.append(chunk_id)
            stored.append(
                StoredChunkRef(
                    chunk_id=chunk_id,
                    section_id=linked_section_id,
                    order_index=chunk.order_index,
                )
            )

        if len(embedding_dims) > 1:
            errors.append(f"inconsistent embedding dimensions: {sorted(embedding_dims)}")
        if embedding_dims and self.expected_embedding_dimension > 0 and self.expected_embedding_dimension not in embedding_dims:
            warnings.append(
                "embedding dimension differs from configured EMBEDDING_DIMENSION "
                f"(expected={self.expected_embedding_dimension}, got={sorted(embedding_dims)})"
            )

        if errors:
            self._log_validation_failure(stage="store_chunks", errors=errors, warnings=warnings)
            raise ValueError(f"Mock ingestion chunks validation failed: {'; '.join(errors)}")

        log.info(
            "Mock ingestion validation passed | stage=store_chunks | doc_id=%s | chunks=%s | unmatched_section_links=%s | warnings=%s",
            document.document_id,
            len(stored),
            unmatched_sections,
            warnings,
        )
        return stored

    def add_rules(
        self,
        document: StoredGuidelineDocument,
        rules: list[RuleCandidate],
    ) -> list[StoredRuleRef]:
        state = self._require_document_state(document.document_id)
        errors: list[str] = []
        warnings: list[str] = []
        seen_statement_keys: set[tuple[str, str]] = set()
        stored: list[StoredRuleRef] = []

        for index, rule in enumerate(rules):
            if not rule.topic.strip():
                errors.append(f"rule[{index}] has empty topic")
            if not rule.statement.strip():
                errors.append(f"rule[{index}] has empty statement")

            source_title = (rule.source_section or "").strip().lower()
            linked_section_id = state.section_by_title.get(source_title) if source_title else None
            if source_title and linked_section_id is None:
                warnings.append(f"rule[{index}] references unknown source_section='{source_title}'")

            statement_key = (rule.statement.strip().lower(), source_title)
            if statement_key in seen_statement_keys:
                warnings.append(f"rule[{index}] duplicates statement for section='{source_title or 'unknown'}'")
            seen_statement_keys.add(statement_key)

            rule_id = uuid.uuid4()
            state.rule_ids.append(rule_id)
            stored.append(StoredRuleRef(rule_id=rule_id, section_id=linked_section_id))

        if errors:
            self._log_validation_failure(stage="store_rules", errors=errors, warnings=warnings)
            raise ValueError(f"Mock ingestion rules validation failed: {'; '.join(errors)}")

        log.info(
            "Mock ingestion validation passed | stage=store_rules | doc_id=%s | rules=%s | warnings=%s",
            document.document_id,
            len(stored),
            warnings,
        )
        return stored

    def flush(self) -> None:
        log.info("Mock ingestion adapter flush | status=noop")

    def _require_document_state(self, document_id: uuid.UUID) -> _MockDocumentState:
        state = self._documents.get(document_id)
        if state is None:
            raise ValueError(f"Unknown mock document handle: {document_id}")
        return state

    def _validate_and_collect_sections(
        self,
        section: NormalizedSection,
        *,
        parent_level: int,
        errors: list[str],
        warnings: list[str],
    ) -> list[StoredSectionRef]:
        refs: list[StoredSectionRef] = []
        section_path = section.section_title.strip() or "<empty_title>"

        if not section.section_title.strip():
            errors.append(f"section '{section_path}' has empty section_title")
        if not section.raw_text.strip():
            warnings.append(f"section '{section_path}' has empty raw_text")
        if not section.cleaned_text.strip():
            warnings.append(f"section '{section_path}' has empty cleaned_text")
        if section.level < 1:
            errors.append(f"section '{section_path}' has invalid level={section.level}")
        if section.order_index < 0:
            errors.append(f"section '{section_path}' has negative order_index={section.order_index}")
        if section.level <= parent_level:
            warnings.append(
                f"section '{section_path}' level={section.level} is not deeper than parent_level={parent_level}"
            )
        if section.page_from is not None and section.page_to is not None and section.page_to < section.page_from:
            errors.append(
                f"section '{section_path}' has invalid page range page_from={section.page_from}, page_to={section.page_to}"
            )

        section_id = uuid.uuid4()
        refs.append(StoredSectionRef(section_id=section_id, section_title=section.section_title))

        for child in section.subsections:
            refs.extend(
                self._validate_and_collect_sections(
                    child,
                    parent_level=section.level,
                    errors=errors,
                    warnings=warnings,
                )
            )
        return refs

    def _log_validation_failure(self, *, stage: str, errors: list[str], warnings: list[str]) -> None:
        log.error(
            "Mock ingestion validation failed | stage=%s | errors=%s | warnings=%s",
            stage,
            errors,
            warnings,
        )
