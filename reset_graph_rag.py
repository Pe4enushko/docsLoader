from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import weaviate

from graphrag_weaviate.config import Settings
from graphrag_weaviate.logging_utils import setup_logging

# Destructive operation. Edit before running if needed.
LOG_FILE = "logs/reset_graph_rag.log"
CHECKPOINT_FILE = ".graphrag_ingest_checkpoint.json"
COLLECTIONS = [
    "VerdictEvaluation",
    "Recommendation",
    "Chunk",
    "Section",
    "Document",
]


def connect_weaviate(settings: Settings):
    parsed = urlparse(settings.weaviate_url)
    grpc_parsed = urlparse("http://localhost:50051")
    return weaviate.connect_to_custom(
        http_host=parsed.hostname or "localhost",
        http_port=8080,
        http_secure=parsed.scheme == "https",
        grpc_host=grpc_parsed.hostname or "localhost",
        grpc_port=grpc_parsed.port or 50051,
        grpc_secure=grpc_parsed.scheme == "https",
    )


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)
    settings = Settings()
    client = connect_weaviate(settings)
    summary: dict[str, object] = {
        "collections_deleted": [],
        "collections_missing": [],
        "checkpoint_removed": False,
    }

    try:
        for name in COLLECTIONS:
            if client.collections.exists(name):
                client.collections.delete(name)
                summary["collections_deleted"].append(name)
                log.info("Deleted collection: %s", name)
            else:
                summary["collections_missing"].append(name)
                log.info("Collection not found (skip): %s", name)

        checkpoint_path = Path(CHECKPOINT_FILE)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            summary["checkpoint_removed"] = True
            log.info("Removed checkpoint file: %s", checkpoint_path)
        else:
            log.info("Checkpoint file not found (skip): %s", checkpoint_path)

        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
