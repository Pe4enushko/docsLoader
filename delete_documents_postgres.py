from __future__ import annotations

import json
import logging
import os
import sys

from engine.graphrag.postgres_dsn import build_graphrag_postgres_dsn, masked_graphrag_dsn_for_logs
from engine.logging_utils import setup_logging
from engine.postgres import PostgresGraphStore

LOG_FILE = "logs/delete_documents_postgres.log"


def parse_doc_ids() -> list[str]:
    if len(sys.argv) > 1:
        return [x.strip() for x in sys.argv[1:] if x.strip()]
    raw = os.getenv("DOC_IDS", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    doc_ids = parse_doc_ids()
    if not doc_ids:
        raise ValueError("Provide doc_id values via args or DOC_IDS env (comma-separated)")

    dsn = build_graphrag_postgres_dsn()
    log.info("Using GraphRAG Postgres DSN: %s", masked_graphrag_dsn_for_logs(dsn))

    store = PostgresGraphStore(dsn=dsn, embedding_dim=1)
    try:
        summary: dict[str, object] = {"backend": "postgres", "doc_ids": doc_ids, "deleted": []}
        for doc_id in doc_ids:
            row = store.delete_document(doc_id)
            row["doc_id"] = doc_id
            summary["deleted"].append(row)
            log.info("Deleted doc_id=%s details=%s", doc_id, json.dumps(row, ensure_ascii=False))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()

