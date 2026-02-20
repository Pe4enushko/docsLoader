# `reset_graph_rag.py`

Destructive reset utility for GraphRAG storage state.

## Purpose

- Deletes GraphRAG collections from Weaviate:
  - `VerdictEvaluation`
  - `Recommendation`
  - `Chunk`
  - `Section`
  - `Document`
- Removes ingestion checkpoint file.

## Editable constants

- `COLLECTIONS`
- `CHECKPOINT_FILE`
- `LOG_FILE`

## Run

```bash
python3 reset_graph_rag.py
```

## Output

JSON summary:

- `collections_deleted`
- `collections_missing`
- `checkpoint_removed`
