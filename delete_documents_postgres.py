from __future__ import annotations

import json
import logging

from engine.graphrag.postgres_dsn import build_graphrag_postgres_dsn, masked_graphrag_dsn_for_logs
from engine.logging_utils import setup_logging
from engine.postgres import PostgresGraphStore

LOG_FILE = "logs/delete_documents_postgres.log"


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    dsn = build_graphrag_postgres_dsn()
    log.info("Using GraphRAG Postgres DSN: %s", masked_graphrag_dsn_for_logs(dsn))

    store = PostgresGraphStore(dsn=dsn, embedding_dim=1)
    try:
        store.clear_all()
        summary = {
            "backend": "postgres",
            "purged": True,
            "scope": "all_graph_nodes_and_edges",
        }
        log.info("Knowledge graph purge completed for Postgres")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
