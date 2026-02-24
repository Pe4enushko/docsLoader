from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from evaluateVerdict import MANIFEST_PATH, build_row_for_medkard, load_manifest_mkb_index
from graphrag_weaviate.config import Settings
from graphrag_weaviate.llm import AppointmentJudge
from graphrag_weaviate.logging_utils import setup_logging
from graphrag_weaviate.retrieval import RetrievalService
from graphrag_weaviate.storage import WeaviateGraphStore
from medkard_postgres import connect_postgres, ensure_medkard_table, upsert_medkard_rows

SOURCE_JSON_PATH = "../data.json"
LOG_FILE = "logs/run_file_verdict_to_db_pipeline.log"
CONCURRENCY_N = 4


def load_appointments_from_file(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        appointments = payload.get("appointments")
        if isinstance(appointments, list):
            return [item for item in appointments if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Input JSON must be object, appointments wrapper, or array")


def chunked_rows(rows: list[tuple], size: int) -> list[list[tuple]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def evaluate_single_appointment(
    settings: Settings,
    appointment: dict[str, Any],
    manifest_exact: dict[str, str],
    manifest_group: dict[str, str],
) -> tuple:
    # Separate store/judge per task avoids sharing non-thread-safe clients between concurrent workers.
    store = WeaviateGraphStore(settings)
    try:
        retrieval = RetrievalService(store, settings)
        judge = AppointmentJudge(retrieval=retrieval, settings=settings)
        return build_row_for_medkard(
            judge=judge,
            appointment=appointment,
            manifest_exact=manifest_exact,
            manifest_group=manifest_group,
        )
    finally:
        store.close()


async def evaluate_with_limit(
    semaphore: asyncio.Semaphore,
    settings: Settings,
    appointment: dict[str, Any],
    manifest_exact: dict[str, str],
    manifest_group: dict[str, str],
) -> tuple:
    async with semaphore:
        return await asyncio.to_thread(
            evaluate_single_appointment,
            settings,
            appointment,
            manifest_exact,
            manifest_group,
        )


async def evaluate_appointments_async(
    settings: Settings,
    appointments: list[dict[str, Any]],
    manifest_exact: dict[str, str],
    manifest_group: dict[str, str],
    concurrency_n: int,
) -> list[tuple]:
    semaphore = asyncio.Semaphore(max(1, concurrency_n))
    tasks = [
        asyncio.create_task(
            evaluate_with_limit(
                semaphore=semaphore,
                settings=settings,
                appointment=item,
                manifest_exact=manifest_exact,
                manifest_group=manifest_group,
            )
        )
        for item in appointments
    ]
    return await asyncio.gather(*tasks)


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    settings = Settings()
    try:
        manifest_exact, manifest_group = load_manifest_mkb_index(MANIFEST_PATH)
        appointments = load_appointments_from_file(SOURCE_JSON_PATH)
        rows = asyncio.run(
            evaluate_appointments_async(
                settings=settings,
                appointments=appointments,
                manifest_exact=manifest_exact,
                manifest_group=manifest_group,
                concurrency_n=CONCURRENCY_N,
            )
        )

        with connect_postgres() as conn:
            ensure_medkard_table(conn)
            upserted = 0
            for batch in chunked_rows(rows, max(1, CONCURRENCY_N)):
                upsert_medkard_rows(conn, batch)
                conn.commit()
                upserted += len(batch)

        print(json.dumps({"source": SOURCE_JSON_PATH, "processed": len(appointments), "upserted": upserted, "db_write": True}, ensure_ascii=False, indent=2))
    except Exception:
        log.exception("File-to-DB verdict pipeline failed")
        raise


if __name__ == "__main__":
    main()
