-- 002_convert_legacy_varchar_to_text.sql
-- Converts legacy Alembic-created VARCHAR columns to TEXT, normalizes check
-- constraints to use LOWER(..), removes deprecated normative DB layer, and
-- enforces UUID for processing_jobs.entity_id.

ALTER TABLE IF EXISTS guideline_documents
    ALTER COLUMN title TYPE text,
    ALTER COLUMN source_path TYPE text,
    ALTER COLUMN checksum TYPE text,
    ALTER COLUMN icd10_codes TYPE text[] USING icd10_codes::text[],
    ALTER COLUMN age_group TYPE text,
    ALTER COLUMN developer TYPE text;

ALTER TABLE IF EXISTS guideline_sections
    ALTER COLUMN section_title TYPE text;

ALTER TABLE IF EXISTS guideline_rules
    ALTER COLUMN topic TYPE text,
    ALTER COLUMN population TYPE text,
    ALTER COLUMN specialty TYPE text,
    ALTER COLUMN conditions TYPE text[] USING conditions::text[],
    ALTER COLUMN triggers TYPE text[] USING triggers::text[],
    ALTER COLUMN audit_targets TYPE text[] USING audit_targets::text[],
    ALTER COLUMN source_section TYPE text;

ALTER TABLE IF EXISTS visit_records
    ALTER COLUMN external_id TYPE text,
    ALTER COLUMN specialty TYPE text,
    ALTER COLUMN icd10_codes TYPE text[] USING icd10_codes::text[],
    ALTER COLUMN extraction_flags TYPE text[] USING extraction_flags::text[];

ALTER TABLE IF EXISTS audit_reports
    ALTER COLUMN status TYPE text;

ALTER TABLE IF EXISTS llm_check_history
    ALTER COLUMN stage TYPE text,
    ALTER COLUMN prompt_version TYPE text,
    ALTER COLUMN model TYPE text,
    ALTER COLUMN status TYPE text;

ALTER TABLE IF EXISTS processing_jobs
    ALTER COLUMN job_type TYPE text,
    ALTER COLUMN status TYPE text;

ALTER TABLE IF EXISTS guideline_documents
    DROP CONSTRAINT IF EXISTS ck_guideline_documents_status,
    ADD CONSTRAINT ck_guideline_documents_status CHECK (LOWER(status) IN ('draft', 'active', 'archived'));

ALTER TABLE IF EXISTS guideline_sections
    DROP CONSTRAINT IF EXISTS ck_guideline_sections_section_type,
    ADD CONSTRAINT ck_guideline_sections_section_type CHECK (
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
    );

ALTER TABLE IF EXISTS guideline_chunks
    DROP CONSTRAINT IF EXISTS ck_guideline_chunks_chunk_type,
    ADD CONSTRAINT ck_guideline_chunks_chunk_type CHECK (
        LOWER(chunk_type) IN ('narrative', 'recommendation', 'table', 'algorithm', 'appendix')
    );

ALTER TABLE IF EXISTS guideline_rules
    DROP CONSTRAINT IF EXISTS ck_guideline_rules_rule_type,
    ADD CONSTRAINT ck_guideline_rules_rule_type CHECK (
        LOWER(rule_type) IN ('diagnostic', 'management', 'followup', 'documentation', 'quality', 'other')
    );

ALTER TABLE IF EXISTS visit_records
    DROP CONSTRAINT IF EXISTS ck_visit_records_visit_type,
    ADD CONSTRAINT ck_visit_records_visit_type CHECK (
        LOWER(visit_type) IN ('primary', 'repeat', 'prophylactic', 'unknown')
    );

DROP TABLE IF EXISTS normative_rules;
DROP TABLE IF EXISTS normative_sections;
DROP TABLE IF EXISTS normative_documents;

DO $$
DECLARE
    invalid_count bigint;
    needs_cast boolean;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'processing_jobs'
          AND column_name = 'entity_id'
          AND data_type <> 'uuid'
    )
    INTO needs_cast;

    IF NOT needs_cast THEN
        RETURN;
    END IF;

    EXECUTE $q$
        SELECT count(*)
        FROM processing_jobs
        WHERE entity_id IS NULL
           OR entity_id::text !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    $q$
    INTO invalid_count;

    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'Cannot cast processing_jobs.entity_id to uuid: % invalid rows', invalid_count;
    END IF;

    EXECUTE 'ALTER TABLE processing_jobs ALTER COLUMN entity_id TYPE uuid USING entity_id::uuid';
END $$;
