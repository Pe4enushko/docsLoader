# `evaluateVerdict.py`

Main production pipeline for MedKard evaluation.

## Purpose

- Fetches appointment items from 1C HTTP endpoint (`appointments` array).
- Evaluates each item with LLM (base + optional KG pass by MKB/doc_id mapping).
- Saves all results to Postgres table `public."MedKard"`.

## Main env inputs

- `ONE_C_APPOINTMENTS_URL`
- `ONE_C_LOGIN`
- `ONE_C_PASSWORD`
- `MANIFEST_PATH`
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_SSLMODE`
- `CONTEXT_TARGET`
- `SCORES_SYSTEM_PROMPT`

## Run

```bash
python3 evaluateVerdict.py
```
