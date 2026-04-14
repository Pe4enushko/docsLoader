-- 001_init_schema.sql
-- Baseline schema for medical audit platform.
-- Policy:
-- - all identifiers are UUID (native PostgreSQL uuid type),
-- - all free-form strings use TEXT instead of VARCHAR.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS guideline_documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title text NOT NULL,
    source_path text NOT NULL,
    checksum text NOT NULL UNIQUE,
    icd10_codes text[] NOT NULL DEFAULT '{}'::text[],
    age_group text,
    publication_year integer,
    developer text,
    status text NOT NULL DEFAULT 'draft',
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ck_guideline_documents_status CHECK (LOWER(status) IN ('draft', 'active', 'archived'))
);

CREATE TABLE IF NOT EXISTS guideline_sections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES guideline_documents(id) ON DELETE CASCADE,
    parent_section_id uuid REFERENCES guideline_sections(id) ON DELETE CASCADE,
    section_type text NOT NULL DEFAULT 'unknown',
    section_title text NOT NULL,
    level integer NOT NULL DEFAULT 1,
    order_index integer NOT NULL DEFAULT 0,
    page_from integer,
    page_to integer,
    raw_text text NOT NULL,
    cleaned_text text NOT NULL,
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ck_guideline_sections_section_type CHECK (
        LOWER(section_type) IN (
            'title_page',
            'toc',
            'abbreviations',
            'definitions',
            'brief_info',
            'diagnostics',
            'treatment',
            'rehabilitation',
            'prevention_followup',
            'organization_of_care',
            'additional_info',
            'quality_criteria',
            'bibliography',
            'appendix',
            'unknown'
        )
    )
);

CREATE TABLE IF NOT EXISTS guideline_chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES guideline_documents(id) ON DELETE CASCADE,
    section_id uuid REFERENCES guideline_sections(id) ON DELETE SET NULL,
    chunk_type text NOT NULL DEFAULT 'narrative',
    chunk_text text NOT NULL,
    token_count integer NOT NULL DEFAULT 0,
    order_index integer NOT NULL DEFAULT 0,
    embedding vector(1024),
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ck_guideline_chunks_chunk_type CHECK (
        LOWER(chunk_type) IN ('narrative', 'recommendation', 'table', 'algorithm', 'appendix')
    )
);

CREATE TABLE IF NOT EXISTS guideline_rules (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES guideline_documents(id) ON DELETE CASCADE,
    section_id uuid REFERENCES guideline_sections(id) ON DELETE SET NULL,
    chunk_id uuid REFERENCES guideline_chunks(id) ON DELETE SET NULL,
    topic text NOT NULL,
    population text,
    specialty text,
    rule_type text NOT NULL DEFAULT 'other',
    statement text NOT NULL,
    conditions text[] NOT NULL DEFAULT '{}'::text[],
    triggers text[] NOT NULL DEFAULT '{}'::text[],
    audit_targets text[] NOT NULL DEFAULT '{}'::text[],
    source_quote text,
    source_section text,
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ck_guideline_rules_rule_type CHECK (
        LOWER(rule_type) IN ('diagnostic', 'management', 'followup', 'documentation', 'quality', 'other')
    )
);

CREATE TABLE IF NOT EXISTS visit_records (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id text UNIQUE,
    raw_json jsonb NOT NULL,
    normalized_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    readable_render text,
    visit_type text NOT NULL DEFAULT 'unknown',
    specialty text,
    patient_age integer,
    icd10_codes text[] NOT NULL DEFAULT '{}'::text[],
    extraction_flags text[] NOT NULL DEFAULT '{}'::text[],
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ck_visit_records_visit_type CHECK (LOWER(visit_type) IN ('primary', 'repeat', 'prophylactic', 'unknown'))
);

CREATE TABLE IF NOT EXISTS audit_reports (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    visit_id uuid NOT NULL REFERENCES visit_records(id) ON DELETE CASCADE,
    report_json jsonb NOT NULL,
    report_text text NOT NULL,
    status text NOT NULL,
    scores_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    llm_trace_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    readable_visit_card text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_check_history (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    visit_id uuid NOT NULL REFERENCES visit_records(id) ON DELETE CASCADE,
    stage text NOT NULL,
    prompt_version text NOT NULL,
    model text NOT NULL,
    input_payload jsonb NOT NULL,
    output_payload jsonb NOT NULL,
    latency_ms integer NOT NULL DEFAULT 0,
    token_usage_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS processing_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type text NOT NULL,
    entity_id uuid NOT NULL,
    status text NOT NULL,
    retries integer NOT NULL DEFAULT 0,
    error_message text,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_guideline_sections_document ON guideline_sections(document_id);
CREATE INDEX IF NOT EXISTS ix_guideline_chunks_document ON guideline_chunks(document_id);
CREATE INDEX IF NOT EXISTS ix_guideline_rules_document ON guideline_rules(document_id);
CREATE INDEX IF NOT EXISTS ix_visit_records_created ON visit_records(created_at);
CREATE INDEX IF NOT EXISTS ix_audit_reports_visit ON audit_reports(visit_id);
