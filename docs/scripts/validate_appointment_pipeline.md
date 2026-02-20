# `validate_appointment_pipeline.py`

Structured quality-evaluation pipeline for full API appointment payloads.

## Purpose

- Loads an API input JSON.
- Retrieves guideline context from GraphRAG by `DOC_ID_OR_MKB_CODE`.
- Calls LLM with structured output schema (`ApiJudgeOutput`).
- Prints combined output: input, retrieved chunks, and final evaluation.

## Editable constants

- `DOC_ID_OR_MKB_CODE`
- `API_INPUT_JSON_PATH`
- `CONTEXT_TARGET`
- `LOG_FILE`

## Run

```bash
python3 validate_appointment_pipeline.py
```

## Output

JSON with:

- `doc_id`
- `query`
- `api_input`
- `retrieved_chunks`
- `evaluation`
