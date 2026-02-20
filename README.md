# GraphRAG Weaviate Library

Python library for document-scoped GraphRAG retrieval and verdict checking using Weaviate and LangChain Ollama.

Detailed setup and workflow docs: `docs/USAGE.md`.

## CLI

- `python docsToGraphRAG.py ingest --input ./pdfs --manifest manifest.json`
- `python docsToGraphRAG.py query --doc_id 10_5 --text "..."`
- `python docsToGraphRAG.py judge --doc_id 10_5 --verdict "..."`

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

Use `tools/visit_validator/validate_visit.py` to send visit data to Ollama, retrieve graph context (strict `doc_id`), and validate diagnosis/recommendations/verdict.

- `python3 tools/visit_validator/validate_visit.py --doc_id 10_5 --query "..." `
- `python3 tools/visit_validator/validate_visit.py --doc_id 10_5 --visit_json ./visit.json`
