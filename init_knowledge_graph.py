from __future__ import annotations

import json
import logging

from graphrag_weaviate.config import Settings
from graphrag_weaviate.ingestion import IngestionService
from graphrag_weaviate.logging_utils import setup_logging
from graphrag_weaviate.storage import WeaviateGraphStore

# Edit values below before running.
INPUT_PDF_DIR = "./pdfs"
MANIFEST_PATH = "./manifest.csv"  # Supports .csv and .json
CHECKPOINT_FILE = ".graphrag_ingest_checkpoint.json"
LOG_FILE = "logs/init_knowledge_graph.log"


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)
    settings = Settings()
    settings.ingest_checkpoint_file = CHECKPOINT_FILE

    store = WeaviateGraphStore(settings)
    try:
        store.init_schema()
        ingester = IngestionService(store, checkpoint_file=settings.ingest_checkpoint_file)
        summary = ingester.ingest(input_dir=INPUT_PDF_DIR, manifest_path=MANIFEST_PATH)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        log.exception("Knowledge graph initialization failed")
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
