from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Enum, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain.enums import VisitType
from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class VisitRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Raw and normalized representation of one outpatient visit.

    Lifecycle:
    - created with `raw_json` at pipeline start,
    - enriched after preprocessing (canonical JSON, flags, classification),
    - linked with one or many generated audit reports.
    """

    __tablename__ = "visit_records"

    # Stable id from external source (for 1C: Прием.GUID). Used for deduplication.
    external_id: Mapped[str | None] = mapped_column(Text, unique=True)

    # Original payload as received from source system; immutable audit artifact.
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Canonicalized visit structure (`meta/patient/subjective/objective/assessment/plan/admin`).
    normalized_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    # Human-readable markdown rendering of normalized card for physician review.
    readable_render: Mapped[str | None] = mapped_column(Text)

    # Visit type predicted from service wording and clinical structure.
    visit_type: Mapped[VisitType] = mapped_column(
        Enum(VisitType, name="visit_type", native_enum=False),
        default=VisitType.UNKNOWN,
        nullable=False,
    )
    # Specialty hint for retrieval and stage prompts (e.g. pediatrics, cardiology).
    specialty: Mapped[str | None] = mapped_column(Text)
    # Parsed patient age used for guideline filters and report context.
    patient_age: Mapped[int | None] = mapped_column(Integer)
    # ICD-10 codes extracted from normalized assessment section.
    icd10_codes: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)

    # Deterministic preprocessing flags (`missing_required_field`, `dx_too_general`, etc.).
    extraction_flags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)

    # Auxiliary metadata from deterministic preprocessing (confidence, reasons, source details).
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    # One visit can produce multiple reports (reruns, prompt/model revisions).
    reports: Mapped[list[AuditReport]] = relationship(back_populates="visit", cascade="all, delete-orphan")
    # Full trace of stage-by-stage LLM calls for reproducibility and debugging.
    llm_checks: Mapped[list[LLMCheckHistory]] = relationship(back_populates="visit", cascade="all, delete-orphan")


class AuditReport(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Final merged audit result for one visit processing run."""

    __tablename__ = "audit_reports"

    # Parent visit being audited.
    visit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("visit_records.id", ondelete="CASCADE"), nullable=False)

    # Machine-readable report: section findings, scores, human review requirement.
    report_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    # Human-readable report text/markdown for manual review workflows.
    report_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Pipeline status snapshot for this report (`ready/partial/failed`).
    status: Mapped[str] = mapped_column(Text, nullable=False)
    # Aggregated numeric metrics useful for dashboards and thresholds.
    scores_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    # Compact LLM trace copy for quick report-level diagnostics.
    llm_trace_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    # Canonical visit card snapshot stored together with report for reviewers.
    readable_visit_card: Mapped[str | None] = mapped_column(Text)

    visit: Mapped[VisitRecord] = relationship(back_populates="reports")


class LLMCheckHistory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Per-stage LLM invocation log (one row per stage per run)."""

    __tablename__ = "llm_check_history"

    # Parent visit of this stage run.
    visit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("visit_records.id", ondelete="CASCADE"), nullable=False)

    # Stage id (`formal_structure_check`, `diagnosis_consistency_check`, ...).
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    # Prompt template version for reproducibility.
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    # Effective LLM model name used for this stage.
    model: Mapped[str] = mapped_column(Text, nullable=False)
    # Input prompts payload sent to model.
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    # Raw model response payload (including provider-specific metadata).
    output_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    # End-to-end stage call latency in milliseconds.
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Tokens/cost counters returned by provider adapter.
    token_usage_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    # Parsed stage status (`ok`, `warning`, `error`, `parse_error`, etc.).
    status: Mapped[str] = mapped_column(Text, nullable=False)

    visit: Mapped[VisitRecord] = relationship(back_populates="llm_checks")


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Generic async/queued processing tracker (ingestion/audit jobs)."""

    __tablename__ = "processing_jobs"

    # Job kind (document ingestion, visit audit, etc.).
    job_type: Mapped[str] = mapped_column(Text, nullable=False)
    # Referenced entity id (document/visit/report).
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    # Processing status in queue lifecycle.
    status: Mapped[str] = mapped_column(Text, nullable=False)
    # Retry counter for transient failures.
    retries: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Last error message (if failed).
    error_message: Mapped[str | None] = mapped_column(Text)
    # Free-form payload with scheduler/debug metadata.
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
