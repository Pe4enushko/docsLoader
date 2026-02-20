# Pipeline Usage (RU Clinic, MKB)

This product validates appointment data quality in a Russian clinic workflow.
The internal GraphRAG layer retrieves recommendation fragments from a knowledge graph indexed by MKB code.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Configure `.env`

```env
WEAVIATE_URL=http://localhost:8080
OLLAMA_EMBED_BASE_URL=http://localhost:11434
OLLAMA_CHAT_BASE_URL=http://localhost:11434

OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=llama3.1:8b

INGEST_CHECKPOINT_FILE=.graphrag_ingest_checkpoint.json
```

## 3. Prepare top-level scripts

Edit constants directly inside scripts before running.

- `evaluateVerdict.py` (judge-only)
  - `DOC_ID_OR_MKB_CODE`
  - `VERDICT_TEXT`
  - `LOG_FILE`
  - Details: `docs/scripts/evaluate_verdict.md`

- `init_knowledge_graph.py`
  - `INPUT_PDF_DIR`
  - `MANIFEST_PATH` (`.json` or `.csv`)
  - `CHECKPOINT_FILE`
  - `LOG_FILE`
  - Details: `docs/scripts/init_knowledge_graph.md`

CSV manifest notes:

- Ingestion iterates manifest records and first tries PDF `<doc_id>.pdf` (for example `1021_1.pdf`).
- If `filename`/`file` is present, it is also accepted as a fallback matcher.
- Manifest key name is also accepted as a fallback (`<key>` or `<key>.pdf`).
- Supported CSV metadata aliases:
  - `ID` -> `doc_id`
  - `Наименование` -> `title`
  - `doc_id`, `title`, `year`, `specialty`, `source_url` are also accepted directly.
- All manifest row fields are stored on `Document.metadata_json` (JSON string).

- `validate_appointment_pipeline.py`
  - `DOC_ID_OR_MKB_CODE`
  - `API_INPUT_JSON_PATH`
  - `CONTEXT_TARGET`
  - `LOG_FILE`
  - Details: `docs/scripts/validate_appointment_pipeline.md`

- `reset_graph_rag.py` (destructive reset utility)
  - `COLLECTIONS`
  - `CHECKPOINT_FILE`
  - `LOG_FILE`
  - Details: `docs/scripts/reset_graph_rag.md`

## 4. Run ingestion

```bash
python3 init_knowledge_graph.py
```

## 5. Run appointment validation

```bash
python3 validate_appointment_pipeline.py
```

## Optional: full GraphRAG reset

Deletes all GraphRAG collections in Weaviate and removes ingestion checkpoint file.

```bash
python3 reset_graph_rag.py
```

Output contract (`evaluation`) is strictly structured via LangChain `with_structured_output`:

```json
{
  "overall_score": 1,
  "risk_level": "low",
  "score_visit_identification": 1,
  "score_anamnesis": 1,
  "score_inspection": 1,
  "score_dynamic": 1,
  "score_diagnosis": 1,
  "score_recommendations": 1,
  "score_structure": 1,
  "issues": "Замечание 1; Замечание 2",
  "summary": "Краткий вывод для главного врача."
}
```

## 6. Main program (judge-only)

```bash
python3 evaluateVerdict.py
```

## 7. Logging behavior

Each run writes logs to terminal and file simultaneously.

- Weaviate operations are logged (upserts/search/evaluation storage).
- LLM answers are logged in truncated form to avoid terminal flooding.
- Errors are logged with stack trace in both outputs.
