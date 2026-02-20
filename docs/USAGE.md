# Pipeline Usage (RU Clinic, MKB)

This product validates appointment data quality in a Russian clinic workflow.
The internal GraphRAG layer retrieves clinical recommendation fragments from a knowledge graph indexed by MKB code.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Configure `.env`

```env
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=

OLLAMA_EMBED_BASE_URL=http://localhost:11434
OLLAMA_CHAT_BASE_URL=http://localhost:11434

OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=llama3.1:8b

INGEST_CHECKPOINT_FILE=.graphrag_ingest_checkpoint.json
```

Notes:
- `OLLAMA_EMBED_BASE_URL` is used only for embeddings.
- `OLLAMA_CHAT_BASE_URL` is used only for chat/judging.
- `OLLAMA_BASE_URL` is still supported as fallback for both.

## 3. Manifest format

Example `manifest.json`:

```json
[
  {
    "filename": "guideline_10_5.pdf",
    "doc_id": "I50",
    "title": "Guideline 10.5",
    "year": 2024,
    "specialty": "cardiology",
    "source_url": "https://example.org/guideline_10_5"
  }
]
```

## 4. Ingest PDFs

```bash
python3 docsToGraphRAG.py ingest --input ./pdfs --manifest manifest.json
```

What it does:
- Extracts text by page
- Detects sections (ToC first, heuristics fallback)
- Chunks section-wise (target ~500–1200 tokens)
- Classifies chunk type
- Embeds chunks with Ollama embeddings
- Upserts document/section/chunk to Weaviate
- Stores ingest checkpoint by `doc_id`

## 5. Query within one document

```bash
python3 docsToGraphRAG.py query --mkb_code I50 --text "ACE inhibitor in CHF with CKD"
```

Optional filters:

```bash
python3 docsToGraphRAG.py query \
  --mkb_code I50 \
  --text "beta blockers" \
  --section_prefix "3.4" \
  --chunk_types recommendation,algorithm \
  --page_start 45 --page_end 80
```

Guarantees:
- Retrieval is always hard-filtered by `doc_id`/`mkb_code`
- Context is packed to 6–12 chunks
- Citations include section path and page range

## 6. Judge doctor verdict

```bash
python3 docsToGraphRAG.py judge --mkb_code I50 --verdict "Start ACE inhibitor and recheck creatinine in 2 weeks"
```

Output contains:
- `verdict`: `correct | partially_correct | incorrect | insufficient_info`
- `explanation`
- `citations`
- `missing_info`
- `recommended_action`

## 7. Validate full medical visit payload

Use separate script:

```bash
python3 tools/visit_validator/validate_visit.py --mkb_code I50 --visit_json tools/visit_validator/visit.sample.json
```

Or free text:

```bash
python3 tools/visit_validator/validate_visit.py --mkb_code I50 --query "Patient has CHF, doctor prescribed ACE inhibitor and renal lab monitoring"
```

## 8. Troubleshooting

- If `ModuleNotFoundError` appears, install deps in active venv:
  - `pip install -e .`
- If Weaviate connection fails, verify `WEAVIATE_URL`.
- If Ollama fails, verify:
  - `OLLAMA_EMBED_BASE_URL`
  - `OLLAMA_CHAT_BASE_URL`
  - model names in `.env`.

## 9. Initialize KG from custom path

Use this standalone initializer to ingest every PDF from any folder.

```bash
python3 tools/init_kg/init_knowledge_graph.py --input /custom/path/to/pdfs
```

Recursive scan:

```bash
python3 tools/init_kg/init_knowledge_graph.py --input /custom/path/to/pdfs --recursive
```

With manifest enrichment:

```bash
python3 tools/init_kg/init_knowledge_graph.py --input /custom/path/to/pdfs --manifest manifest.json
```

Strict mode (fail if any PDF has no manifest row):

```bash
python3 tools/init_kg/init_knowledge_graph.py --input /custom/path/to/pdfs --manifest manifest.json --strict
```
