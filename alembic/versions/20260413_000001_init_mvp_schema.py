"""init mvp schema

Revision ID: 20260413_000001
Revises:
Create Date: 2026-04-13 00:00:01
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

try:
    from pgvector.sqlalchemy import Vector
except Exception:  # pragma: no cover
    Vector = None


revision = "20260413_000001"
down_revision = None
branch_labels = None
depends_on = None


def _vector_type() -> sa.types.TypeEngine:
    if Vector is not None:
        return Vector(768)
    return postgresql.ARRAY(sa.Float)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "guideline_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("title", sa.String(length=1024), nullable=False),
        sa.Column("source_path", sa.String(length=2048), nullable=False),
        sa.Column("checksum", sa.String(length=128), nullable=False, unique=True),
        sa.Column("icd10_codes", postgresql.ARRAY(sa.String(length=16)), nullable=False, server_default="{}"),
        sa.Column("age_group", sa.String(length=128), nullable=True),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("developer", sa.String(length=255), nullable=True),
        sa.Column(
            "status",
            sa.Enum("draft", "active", "archived", name="guideline_document_status", native_enum=False),
            nullable=False,
            server_default="draft",
        ),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "guideline_sections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("guideline_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_section_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("guideline_sections.id", ondelete="CASCADE"), nullable=True),
        sa.Column(
            "section_type",
            sa.Enum(
                "title_page",
                "toc",
                "abbreviations",
                "definitions",
                "brief_info",
                "diagnostics",
                "treatment",
                "rehabilitation",
                "prevention_followup",
                "organization_of_care",
                "additional_info",
                "quality_criteria",
                "bibliography",
                "appendix",
                "unknown",
                name="guideline_section_type",
                native_enum=False,
            ),
            nullable=False,
            server_default="unknown",
        ),
        sa.Column("section_title", sa.String(length=1024), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("page_from", sa.Integer(), nullable=True),
        sa.Column("page_to", sa.Integer(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("cleaned_text", sa.Text(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "guideline_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("guideline_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("guideline_sections.id", ondelete="SET NULL"), nullable=True),
        sa.Column(
            "chunk_type",
            sa.Enum(
                "narrative",
                "recommendation",
                "table",
                "algorithm",
                "appendix",
                name="guideline_chunk_type",
                native_enum=False,
            ),
            nullable=False,
            server_default="narrative",
        ),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding", _vector_type(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "guideline_rules",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("guideline_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("guideline_sections.id", ondelete="SET NULL"), nullable=True),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("guideline_chunks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("topic", sa.String(length=256), nullable=False),
        sa.Column("population", sa.String(length=256), nullable=True),
        sa.Column("specialty", sa.String(length=128), nullable=True),
        sa.Column(
            "rule_type",
            sa.Enum(
                "diagnostic",
                "management",
                "followup",
                "documentation",
                "quality",
                "other",
                name="guideline_rule_type",
                native_enum=False,
            ),
            nullable=False,
            server_default="other",
        ),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("conditions", postgresql.ARRAY(sa.String(length=512)), nullable=False, server_default="{}"),
        sa.Column("triggers", postgresql.ARRAY(sa.String(length=512)), nullable=False, server_default="{}"),
        sa.Column("audit_targets", postgresql.ARRAY(sa.String(length=512)), nullable=False, server_default="{}"),
        sa.Column("source_quote", sa.Text(), nullable=True),
        sa.Column("source_section", sa.String(length=1024), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "normative_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("title", sa.String(length=1024), nullable=False),
        sa.Column("source_path", sa.String(length=2048), nullable=False),
        sa.Column("checksum", sa.String(length=128), nullable=False, unique=True),
        sa.Column("issuer", sa.String(length=256), nullable=True),
        sa.Column("effective_date", sa.String(length=64), nullable=True),
        sa.Column(
            "status",
            sa.Enum("draft", "active", "archived", name="normative_document_status", native_enum=False),
            nullable=False,
            server_default="draft",
        ),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "normative_sections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("normative_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_section_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("normative_sections.id", ondelete="CASCADE"), nullable=True),
        sa.Column("section_title", sa.String(length=1024), nullable=False),
        sa.Column(
            "section_type",
            sa.Enum(
                "title_page",
                "toc",
                "abbreviations",
                "definitions",
                "brief_info",
                "diagnostics",
                "treatment",
                "rehabilitation",
                "prevention_followup",
                "organization_of_care",
                "additional_info",
                "quality_criteria",
                "bibliography",
                "appendix",
                "unknown",
                name="normative_section_type",
                native_enum=False,
            ),
            nullable=False,
            server_default="unknown",
        ),
        sa.Column("level", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("cleaned_text", sa.Text(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "normative_rules",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("normative_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("normative_sections.id", ondelete="SET NULL"), nullable=True),
        sa.Column("topic", sa.String(length=256), nullable=False),
        sa.Column(
            "rule_type",
            sa.Enum(
                "diagnostic",
                "management",
                "followup",
                "documentation",
                "quality",
                "other",
                name="normative_rule_type",
                native_enum=False,
            ),
            nullable=False,
            server_default="other",
        ),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("conditions", postgresql.ARRAY(sa.String(length=512)), nullable=False, server_default="{}"),
        sa.Column("audit_targets", postgresql.ARRAY(sa.String(length=512)), nullable=False, server_default="{}"),
        sa.Column("source_quote", sa.Text(), nullable=True),
        sa.Column("source_section", sa.String(length=1024), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "visit_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=True, unique=True),
        sa.Column("raw_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("normalized_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("readable_render", sa.Text(), nullable=True),
        sa.Column(
            "visit_type",
            sa.Enum("primary", "repeat", "prophylactic", "unknown", name="visit_type", native_enum=False),
            nullable=False,
            server_default="unknown",
        ),
        sa.Column("specialty", sa.String(length=128), nullable=True),
        sa.Column("patient_age", sa.Integer(), nullable=True),
        sa.Column("icd10_codes", postgresql.ARRAY(sa.String(length=16)), nullable=False, server_default="{}"),
        sa.Column("extraction_flags", postgresql.ARRAY(sa.String(length=128)), nullable=False, server_default="{}"),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "audit_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("visit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("visit_records.id", ondelete="CASCADE"), nullable=False),
        sa.Column("report_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("report_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("scores_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("llm_trace_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("readable_visit_card", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "llm_check_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("visit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("visit_records.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("input_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("output_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("token_usage_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "processing_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("retries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_guideline_sections_document", "guideline_sections", ["document_id"])
    op.create_index("ix_guideline_chunks_document", "guideline_chunks", ["document_id"])
    op.create_index("ix_guideline_rules_document", "guideline_rules", ["document_id"])
    op.create_index("ix_normative_sections_document", "normative_sections", ["document_id"])
    op.create_index("ix_normative_rules_document", "normative_rules", ["document_id"])
    op.create_index("ix_visit_records_created", "visit_records", ["created_at"])
    op.create_index("ix_audit_reports_visit", "audit_reports", ["visit_id"])
    op.create_index("ix_llm_check_history_visit", "llm_check_history", ["visit_id"])
    op.create_index("ix_processing_jobs_status", "processing_jobs", ["status", "job_type"])


def downgrade() -> None:
    op.drop_index("ix_processing_jobs_status", table_name="processing_jobs")
    op.drop_index("ix_llm_check_history_visit", table_name="llm_check_history")
    op.drop_index("ix_audit_reports_visit", table_name="audit_reports")
    op.drop_index("ix_visit_records_created", table_name="visit_records")
    op.drop_index("ix_normative_rules_document", table_name="normative_rules")
    op.drop_index("ix_normative_sections_document", table_name="normative_sections")
    op.drop_index("ix_guideline_rules_document", table_name="guideline_rules")
    op.drop_index("ix_guideline_chunks_document", table_name="guideline_chunks")
    op.drop_index("ix_guideline_sections_document", table_name="guideline_sections")

    op.drop_table("processing_jobs")
    op.drop_table("llm_check_history")
    op.drop_table("audit_reports")
    op.drop_table("visit_records")
    op.drop_table("normative_rules")
    op.drop_table("normative_sections")
    op.drop_table("normative_documents")
    op.drop_table("guideline_rules")
    op.drop_table("guideline_chunks")
    op.drop_table("guideline_sections")
    op.drop_table("guideline_documents")

    op.execute("DROP TYPE IF EXISTS normative_rule_type")
    op.execute("DROP TYPE IF EXISTS normative_section_type")
    op.execute("DROP TYPE IF EXISTS normative_document_status")
    op.execute("DROP TYPE IF EXISTS guideline_rule_type")
    op.execute("DROP TYPE IF EXISTS guideline_chunk_type")
    op.execute("DROP TYPE IF EXISTS guideline_section_type")
    op.execute("DROP TYPE IF EXISTS guideline_document_status")
    op.execute("DROP TYPE IF EXISTS visit_type")
