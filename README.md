# Appointment Validation Pipeline (RU Clinic)

This repository is the product pipeline for validating how medics fill appointment data in a Russian clinic.
Validation is grounded in a knowledge graph built from clinical recommendations indexed by MKB codes.

Detailed setup: `docs/USAGE.md`.
Human-readable overview: `docs/HOW_IT_WORKS.md`.

## Top-Level Scripts

- `docsToGraphRAG.py`
  - Main production entrypoint. Judge-only execution.
- `validate_appointment_pipeline.py`
  - Extended structured API-payload validation flow (used for integration/testing scenarios).
- `init_knowledge_graph.py`
  - Knowledge graph ingestion helper (run separately, typically for setup/testing).

## Environment

Values are loaded from `.env`:

- `WEAVIATE_URL`
- `OLLAMA_EMBED_BASE_URL`
- `OLLAMA_CHAT_BASE_URL`
- `OLLAMA_EMBED_MODEL`
- `OLLAMA_CHAT_MODEL`
- `INGEST_CHECKPOINT_FILE`

## Logging

All scripts log simultaneously to:

- terminal (`stdout` via `StreamHandler`)
- file (`logs/*.log` via `FileHandler`)

Core modules also log:

- Weaviate actions (connect/create collection/upsert/search/store evaluation)
- retrieval stages and counts
- LLM raw answers (truncated to avoid flooding output)
