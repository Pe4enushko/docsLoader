from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Enum, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.domain.enums import ChunkType, DocumentStatus, RuleType, SectionType
from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.types import vector_type

EMBEDDING_VECTOR_DIM = get_settings().embedding_dimension


class GuidelineDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Clinical guideline source document metadata and ingestion status."""

    __tablename__ = "guideline_documents"

    # Human-readable title of guideline.
    title: Mapped[str] = mapped_column(Text, nullable=False)
    # Original file path used for ingestion.
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    # File checksum for deduplication and idempotent ingestion.
    checksum: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    # ICD-10 codes associated with this guideline.
    icd10_codes: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    # Age segment from title page metadata.
    age_group: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Publication/update year when available.
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Organization/authoring body.
    developer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Ingestion lifecycle status.
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="guideline_document_status", native_enum=False),
        default=DocumentStatus.DRAFT,
        nullable=False,
    )

    # Extra parser metadata (title page, parser diagnostics, etc.).
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    sections: Mapped[list[GuidelineSection]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[list[GuidelineChunk]] = relationship(back_populates="document", cascade="all, delete-orphan")
    rules: Mapped[list[GuidelineRule]] = relationship(back_populates="document", cascade="all, delete-orphan")


class GuidelineSection(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Normalized hierarchical section/subsection of a clinical guideline."""

    __tablename__ = "guideline_sections"

    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("guideline_documents.id", ondelete="CASCADE"), nullable=False)
    parent_section_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("guideline_sections.id", ondelete="CASCADE"))

    section_type: Mapped[SectionType] = mapped_column(
        Enum(SectionType, name="guideline_section_type", native_enum=False),
        default=SectionType.UNKNOWN,
        nullable=False,
    )
    section_title: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_from: Mapped[int | None] = mapped_column(Integer)
    page_to: Mapped[int | None] = mapped_column(Integer)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    document: Mapped[GuidelineDocument] = relationship(back_populates="sections")
    parent: Mapped[GuidelineSection | None] = relationship(remote_side="GuidelineSection.id", back_populates="children")
    children: Mapped[list[GuidelineSection]] = relationship(back_populates="parent", cascade="all, delete-orphan")

    chunks: Mapped[list[GuidelineChunk]] = relationship(back_populates="section")
    rules: Mapped[list[GuidelineRule]] = relationship(back_populates="section")


class GuidelineChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """RAG-ready text chunk linked to guideline document/section."""

    __tablename__ = "guideline_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("guideline_documents.id", ondelete="CASCADE"), nullable=False)
    section_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("guideline_sections.id", ondelete="SET NULL"))

    chunk_type: Mapped[ChunkType] = mapped_column(
        Enum(ChunkType, name="guideline_chunk_type", native_enum=False),
        default=ChunkType.NARRATIVE,
        nullable=False,
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Must match embedding provider output dimension (configured via EMBEDDING_DIMENSION).
    embedding: Mapped[list[float] | None] = mapped_column(vector_type(EMBEDDING_VECTOR_DIM))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    document: Mapped[GuidelineDocument] = relationship(back_populates="chunks")
    section: Mapped[GuidelineSection | None] = relationship(back_populates="chunks")
    rules: Mapped[list[GuidelineRule]] = relationship(back_populates="chunk")


class GuidelineRule(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Structured rule extracted from guideline text (LLM-assisted + heuristic)."""

    __tablename__ = "guideline_rules"

    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("guideline_documents.id", ondelete="CASCADE"), nullable=False)
    section_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("guideline_sections.id", ondelete="SET NULL"))
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("guideline_chunks.id", ondelete="SET NULL"))

    topic: Mapped[str] = mapped_column(Text, nullable=False)
    population: Mapped[str | None] = mapped_column(Text)
    specialty: Mapped[str | None] = mapped_column(Text)
    rule_type: Mapped[RuleType] = mapped_column(
        Enum(RuleType, name="guideline_rule_type", native_enum=False),
        default=RuleType.OTHER,
        nullable=False,
    )

    statement: Mapped[str] = mapped_column(Text, nullable=False)
    conditions: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    triggers: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    audit_targets: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)

    source_quote: Mapped[str | None] = mapped_column(Text)
    source_section: Mapped[str | None] = mapped_column(Text)

    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    document: Mapped[GuidelineDocument] = relationship(back_populates="rules")
    section: Mapped[GuidelineSection | None] = relationship(back_populates="rules")
    chunk: Mapped[GuidelineChunk | None] = relationship(back_populates="rules")


class NormativeDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Normative/legal document metadata used by audit rules layer."""

    __tablename__ = "normative_documents"

    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    issuer: Mapped[str | None] = mapped_column(Text)
    effective_date: Mapped[str | None] = mapped_column(Text)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="normative_document_status", native_enum=False),
        default=DocumentStatus.DRAFT,
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    sections: Mapped[list[NormativeSection]] = relationship(back_populates="document", cascade="all, delete-orphan")
    rules: Mapped[list[NormativeRule]] = relationship(back_populates="document", cascade="all, delete-orphan")


class NormativeSection(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Normalized section hierarchy for normative documents."""

    __tablename__ = "normative_sections"

    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("normative_documents.id", ondelete="CASCADE"), nullable=False)
    parent_section_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("normative_sections.id", ondelete="CASCADE"))

    section_title: Mapped[str] = mapped_column(Text, nullable=False)
    section_type: Mapped[SectionType] = mapped_column(
        Enum(SectionType, name="normative_section_type", native_enum=False),
        default=SectionType.UNKNOWN,
        nullable=False,
    )
    level: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    document: Mapped[NormativeDocument] = relationship(back_populates="sections")
    parent: Mapped[NormativeSection | None] = relationship(remote_side="NormativeSection.id", back_populates="children")
    children: Mapped[list[NormativeSection]] = relationship(back_populates="parent", cascade="all, delete-orphan")

    rules: Mapped[list[NormativeRule]] = relationship(back_populates="section")


class NormativeRule(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Structured compliance rule extracted from normative sections."""

    __tablename__ = "normative_rules"

    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("normative_documents.id", ondelete="CASCADE"), nullable=False)
    section_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("normative_sections.id", ondelete="SET NULL"))

    topic: Mapped[str] = mapped_column(Text, nullable=False)
    rule_type: Mapped[RuleType] = mapped_column(
        Enum(RuleType, name="normative_rule_type", native_enum=False),
        default=RuleType.OTHER,
        nullable=False,
    )
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    conditions: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    audit_targets: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    source_quote: Mapped[str | None] = mapped_column(Text)
    source_section: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    document: Mapped[NormativeDocument] = relationship(back_populates="rules")
    section: Mapped[NormativeSection | None] = relationship(back_populates="rules")
