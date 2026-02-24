from __future__ import annotations

import json
import logging
import os
import sys

from weaviate.classes.query import Filter

from engine.config import Settings
from engine.logging_utils import setup_logging
from engine.weaviate import WeaviateGraphStore

LOG_FILE = "logs/delete_documents_weaviate.log"


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

    settings = Settings()
    store = WeaviateGraphStore(settings)
    try:
        collections = [store.EVALS, store.RECS, store.CHUNKS, store.SECTIONS, store.DOCS]
        summary: dict[str, object] = {"backend": "weaviate", "doc_ids": doc_ids, "deleted": []}
        for doc_id in doc_ids:
            per_doc: dict[str, object] = {"doc_id": doc_id, "collections": {}}
            for name in collections:
                collection = store.client.collections.get(name)
                result = collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
                per_doc["collections"][name] = str(result)
                log.info("Delete collection=%s doc_id=%s result=%s", name, doc_id, result)
            summary["deleted"].append(per_doc)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()

