from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain.enums import VisitType
from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class VisitRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "visit_records"

    external_id: Mapped[str | None] = mapped_column(String(128), unique=True)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    normalized_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    readable_render: Mapped[str | None] = mapped_column(Text)

    visit_type: Mapped[VisitType] = mapped_column(
        Enum(VisitType, name="visit_type", native_enum=False),
        default=VisitType.UNKNOWN,
        nullable=False,
    )
    specialty: Mapped[str | None] = mapped_column(String(128))
    patient_age: Mapped[int | None] = mapped_column(Integer)
    icd10_codes: Mapped[list[str]] = mapped_column(ARRAY(String(16)), default=list, nullable=False)

    extraction_flags: Mapped[list[str]] = mapped_column(ARRAY(String(128)), default=list, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    reports: Mapped[list[AuditReport]] = relationship(back_populates="visit", cascade="all, delete-orphan")
    llm_checks: Mapped[list[LLMCheckHistory]] = relationship(back_populates="visit", cascade="all, delete-orphan")


class AuditReport(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "audit_reports"

    visit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("visit_records.id", ondelete="CASCADE"), nullable=False)

    report_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    report_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    scores_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    llm_trace_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    readable_visit_card: Mapped[str | None] = mapped_column(Text)

    visit: Mapped[VisitRecord] = relationship(back_populates="reports")


class LLMCheckHistory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "llm_check_history"

    visit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("visit_records.id", ondelete="CASCADE"), nullable=False)

    stage: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    output_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    token_usage_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)

    visit: Mapped[VisitRecord] = relationship(back_populates="llm_checks")


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "processing_jobs"

    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    retries: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
