# Appointment Validation Pipeline (RU Clinic)

This repository is the product pipeline for validating how a medic filled appointment data in a Russian clinic.
It uses an internal GraphRAG module with Weaviate + Ollama and a knowledge graph of clinical recommendations grouped by MKB code.

Detailed setup and workflow docs: `docs/USAGE.md`.

## CLI

- `python docsToGraphRAG.py ingest --input ./pdfs --manifest manifest.json`
- `python docsToGraphRAG.py query --mkb_code I50 --text "..."`
- `python docsToGraphRAG.py judge --mkb_code I50 --verdict "..."`

## Required env vars

Values are loaded from `.env` (see `.env.example`).

- `WEAVIATE_URL` (default: `http://localhost:8080`)
- `WEAVIATE_API_KEY` (optional)
- `OLLAMA_EMBED_BASE_URL` (embeddings endpoint)
- `OLLAMA_CHAT_BASE_URL` (chat endpoint)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)
- `OLLAMA_CHAT_MODEL` (default: `llama3.1:8b`)
- `OLLAMA_BASE_URL` (optional fallback for both, backward-compatible)

## Visit Validator Script

Use `tools/visit_validator/validate_visit.py` to send visit data to Ollama, retrieve graph context (strict `doc_id`/`mkb_code`), and validate diagnosis/recommendations/verdict.

- `python3 tools/visit_validator/validate_visit.py --mkb_code I50 --query "..." `
- `python3 tools/visit_validator/validate_visit.py --mkb_code I50 --visit_json ./visit.json`
