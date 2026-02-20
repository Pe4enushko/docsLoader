# `evaluateVerdict.py`

Judge-only script for evaluating one verdict statement against guideline context.

## Purpose

- Retrieves relevant chunks for a single `doc_id`/MKB code.
- Runs verdict judge logic.
- Prints compact JSON result with verdict and citations.

## Editable constants

- `DOC_ID_OR_MKB_CODE`
- `VERDICT_TEXT`
- `CHECKPOINT_FILE`
- `LOG_FILE`

## Run

```bash
python3 evaluateVerdict.py
```

## Output

JSON with:

- `doc_id`
- `verdict`
- `explanation`
- `citations`
- `missing_info`
- `recommended_action`
