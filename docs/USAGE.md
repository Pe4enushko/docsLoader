# Pipeline Usage (RU Clinic, MKB)

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Configure `.env`

Create `.env` from `.env.example` and fill all values.

Main required groups:
- Weaviate/Ollama
- 1C HTTP auth and endpoint
- Postgres connection parts
- `MANIFEST_PATH`

## 3. Run knowledge graph ingestion

```bash
python3 init_knowledge_graph.py
```

## 4. Initialize MedKard table

```bash
python3 init_medkard_table.py
```

## 5. Run main validation pipeline

```bash
python3 evaluateVerdict.py
```

Pipeline behavior:
1. Calls 1C HTTP endpoint and reads `appointments` array.
2. For each appointment, extracts MKB codes if present.
3. Runs base scoring.
4. If MKB exists and matched in `manifest.csv`, runs KG-grounded scoring for that `doc_id` and merges result.
5. Stores full result into Postgres `public."MedKard"`.

## Logging

Each run writes logs to terminal and file.
