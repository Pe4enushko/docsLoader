# Medical Document Intelligence & Visit Audit MVP

Production-oriented starter implementation for:

- clinical guideline ingestion (PDF/DOC/DOCX via Apache Tika),
- normalization into unified section hierarchy,
- RAG-ready chunk/rule storage in PostgreSQL + pgvector,
- visit JSON preprocessing and heuristic quality flags,
- visit type classification (`primary`, `repeat`, `prophylactic`),
- staged LLM checks with retrieval adapter abstraction,
- final report generation and persistence.

## Architecture (high level)

- `app/models` — SQLAlchemy 2.x ORM models
- `app/schemas` — Pydantic contracts for ingestion, retrieval, LLM checks, reports
- `app/repositories` — persistence layer for documents/visits/reports/jobs
- `app/ingestion` — Tika parsing, cleaning, section normalization, chunking, rule extraction
- `app/integrations/one_c` — 1C retrieval client and payload parser
- `app/rag` — retrieval adapter abstraction + PostgreSQL implementation
- `app/prompts` — separate prompt builders per check stage
- `app/services` — visit normalization, flags, classification, diagnosis extraction, rendering, report builder
- `app/pipelines` — orchestration pipelines:
  - `GuidelineIngestionPipeline`
  - `NormativeIngestionPipeline`
  - `VisitAuditPipeline`
- `alembic` — migration environment and initial schema
- `tests` — baseline tests for core contracts

## Database

Initial migration: `alembic/versions/20260413_000001_init_mvp_schema.py`

Main tables:

- `guideline_documents`, `guideline_sections`, `guideline_chunks`, `guideline_rules`
- `normative_documents`, `normative_sections`, `normative_rules`
- `visit_records`, `audit_reports`, `llm_check_history`, `processing_jobs`

`pgvector` support is included via `CREATE EXTENSION IF NOT EXISTS vector` and vector-capable embedding column.

## Quickstart

1. Create env file:

```bash
cp .env.example .env
```

Set at least:

- `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_NAME`
- `OPENAI_API_KEY`
- `OPENAI_LLM_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `LOG_LEVEL` (`INFO` by default, use `DEBUG` for verbose step logs)
- optionally `OPENAI_LLM_BASE_URL` / `OPENAI_EMBEDDING_BASE_URL` for compatible gateways.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run migrations:

```bash
alembic upgrade head
```

4. Run tests:

```bash
pytest
```

## Batch Scripts

Export raw 1C payload by date range to local file:

```bash
python3 scripts/export_1c_payload.py
```

Uses env vars: `ONE_C_DATE_BEGIN`, `ONE_C_DATE_END`, `ONE_C_EXPORT_PATH`.

Run guideline ingestion for all PDFs from env folder with optional cleanup:

```bash
python3 scripts/run_guideline_ingestion.py
```

Uses env vars: `GUIDELINES_DIR`, `GUIDELINES_GLOB`, `GUIDELINES_RECURSIVE`, `INGEST_CLEAR_PREVIOUS`.

Run audit pipeline on test data file and export batch results to JSON:

```bash
python3 scripts/run_test_data_audit.py
```

Uses env vars: `TEST_DATA_INPUT_PATH`, `TEST_DATA_OUTPUT_PATH`, `TEST_DATA_CONTINUE_ON_ERROR`.

## Programmatic usage

```python
from app.app import audit_visit, ingest_all_guidelines_from_env, init_db

init_db()  # optional if you don't use Alembic yet
results = ingest_all_guidelines_from_env()

visit_payload = {
    "patient": {"age": 42},
    "subjective": {"complaints": "кашель"},
    "objective": {"status": "t=37.2"},
    "assessment": {"diagnosis": "J06.9"},
    "plan": {"actions": "контроль через 3 дня"},
}

report_result = audit_visit(visit_payload, external_id="visit-123")
```

## Current status

This is an MVP-ready skeleton designed for extension:

- retrieval and embeddings are abstracted (`RetrievalAdapter`, `EmbeddingProvider`),
- LLM calls and embeddings default to OpenAI-compatible providers (`OpenAILLMClient`, `OpenAIEmbeddingProvider`),
- rule extraction from guidelines is LLM-assisted with heuristic fallback,
- prompts are modular and versioned by stage,
- normalization and extraction use robust heuristics with unknown fallbacks.
