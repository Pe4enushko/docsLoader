from __future__ import annotations

import json
import logging

from engine.graphrag.pgvector_age_adapter import PgvectorAgeAdapter
from engine.graphrag.postgres_dsn import build_graphrag_postgres_dsn, masked_graphrag_dsn_for_logs
from engine.ingestion import IngestionService
from engine.logging_utils import setup_logging

# Edit values below before running.
INPUT_PDF_DIR = "./pdfs"
MANIFEST_PATH = "./manifest.csv"  # Supports .csv and .json
CHECKPOINT_FILE = ".graphrag_ingest_checkpoint_pgvector.json"
LOG_FILE = "logs/init_knowledge_graph_pgvector.log"


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    dsn = build_graphrag_postgres_dsn()
    log.info("Using GraphRAG Postgres DSN: %s", masked_graphrag_dsn_for_logs(dsn))

    adapter = PgvectorAgeAdapter(dsn=dsn)
    try:
        adapter.init_schema()
        ingester = IngestionService(adapter, checkpoint_file=CHECKPOINT_FILE)
        summary = ingester.ingest(input_dir=INPUT_PDF_DIR, manifest_path=MANIFEST_PATH)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        log.exception("Knowledge graph initialization (pgvector) failed")
        raise
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
