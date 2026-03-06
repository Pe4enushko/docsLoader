from __future__ import annotations

import json
import logging

from weaviate.classes.query import Filter

from engine.config import Settings
from engine.logging_utils import setup_logging
from engine.weaviate import WeaviateGraphStore

LOG_FILE = "logs/delete_documents_weaviate.log"


def fetch_all_doc_ids(store: WeaviateGraphStore, page_size: int = 200) -> list[str]:
    docs_collection = store.client.collections.get(store.DOCS)
    out: list[str] = []
    seen: set[str] = set()
    offset = 0

    while True:
        response = docs_collection.query.fetch_objects(limit=page_size, offset=offset)
        objects = response.objects
        if not objects:
            break

        for obj in objects:
            doc_id = str((obj.properties or {}).get("doc_id", "")).strip()
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)

        if len(objects) < page_size:
            break
        offset += page_size

    return out


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    settings = Settings()
    store = WeaviateGraphStore(settings)
    try:
        collections = [store.EVALS, store.RECS, store.CHUNKS, store.SECTIONS, store.DOCS]
        doc_ids = fetch_all_doc_ids(store)

        summary: dict[str, object] = {
            "backend": "weaviate",
            "purged": True,
            "doc_ids_total": len(doc_ids),
            "deleted": [],
        }

        for doc_id in doc_ids:
            per_doc: dict[str, object] = {"doc_id": doc_id, "collections": {}}
            for name in collections:
                collection = store.client.collections.get(name)
                result = collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
                per_doc["collections"][name] = str(result)
                log.info("Purge collection=%s doc_id=%s result=%s", name, doc_id, result)
            summary["deleted"].append(per_doc)

        log.info("Knowledge graph purge completed for Weaviate, doc_ids=%d", len(doc_ids))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
