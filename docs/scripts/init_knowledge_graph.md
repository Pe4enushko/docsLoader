# `init_knowledge_graph.py`

Initial ingestion script for building/updating the GraphRAG knowledge graph from PDF guidelines.

## Purpose

- Reads manifest (`.csv` or `.json`) and resolves PDFs (ID-first: `<doc_id>.pdf`).
- Extracts sections/chunks, computes embeddings, stores graph objects in Weaviate.
- Persists ingestion state to avoid reprocessing already completed documents.

## Editable constants

- `INPUT_PDF_DIR`
- `MANIFEST_PATH`
- `CHECKPOINT_FILE`
- `LOG_FILE`

## State files

- `CHECKPOINT_FILE` (`.graphrag_ingest_checkpoint.json`): JSON map `doc_id -> "done"`.
If a `doc_id` is already marked `done` in checkpoint, ingestion skips it and logs that skip.

## Run

```bash
python3 init_knowledge_graph.py
```
