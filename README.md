# Appointment Validation Pipeline (RU Clinic)

This repository validates how medical appointment records are filled in a Russian clinic workflow.
Validation is grounded in a knowledge graph indexed by MKB codes.

Detailed setup: `docs/USAGE.md`.
Overview: `docs/HOW_IT_WORKS.md`.

## Top-Level Scripts

- `evaluateVerdict.py`
  - Main production pipeline.
  - Fetches appointments from 1C HTTP API, evaluates each item, and writes results to `MedKard` in Postgres.
  - Details: `docs/scripts/evaluate_verdict.md`
- `init_medkard_table.py`
  - Initializes `MedKard` table in Postgres from SQL schema.
- `init_knowledge_graph.py`
  - Ingests guideline PDFs into knowledge graph.
  - Details: `docs/scripts/init_knowledge_graph.md`
- `reset_graph_rag.py`
  - Destructive reset helper for GraphRAG collections/checkpoint.
  - Details: `docs/scripts/reset_graph_rag.md`

## Environment

Use `.env` based on `.env.example`.

## Logging

Scripts log to terminal and `logs/*.log`.
