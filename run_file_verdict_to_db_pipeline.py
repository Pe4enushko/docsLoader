from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from evaluateVerdict import (
    MANIFEST_PATH,
    build_row_for_medkard,
    extract_visit_guid,
    load_manifest_mkb_index,
)
from engine.appointments import parse_appointments_payload
from engine.config import Settings
from engine.llm import AppointmentJudge
from engine.logging_utils import setup_logging
from engine.models import MedKardRow
from engine.retrieval import RetrievalService
from engine.storage import WeaviateGraphStore
from engine.postgres import connect_postgres, ensure_medkard_table, is_visit_processed, upsert_medkard_rows

SOURCE_JSON_PATH = "../data.json"
LOG_FILE = "logs/run_file_verdict_to_db_pipeline.log"
CONCURRENCY_N = 4


def load_appointments_from_file(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_appointments_payload(payload)


def evaluate_single_appointment(
    settings: Settings,
    appointment: dict[str, Any],
    manifest_exact: dict[str, str],
    manifest_group: dict[str, str],
) -> MedKardRow:
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
) -> MedKardRow:
    async with semaphore:
        return await asyncio.to_thread(
            evaluate_single_appointment,
            settings,
            appointment,
            manifest_exact,
            manifest_group,
        )


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    settings = Settings()
    try:
        manifest_exact, manifest_group = load_manifest_mkb_index(MANIFEST_PATH)
        appointments = load_appointments_from_file(SOURCE_JSON_PATH)

        with connect_postgres() as conn:
            ensure_medkard_table(conn)
            scheduled: list[tuple[int, str, dict[str, Any]]] = []
            skipped_missing_guid = 0
            skipped_preprocessed = 0
            seen_guids: set[str] = set()
            for idx, appointment in enumerate(appointments):
                visit_guid = extract_visit_guid(appointment)
                if not visit_guid:
                    skipped_missing_guid += 1
                    log.error("Skipping appointment before evaluation index=%d reason=missing_guid", idx)
                    continue
                if visit_guid in seen_guids:
                    skipped_preprocessed += 1
                    log.info("Skipping duplicate GUID in input index=%d visit_guid=%s", idx, visit_guid)
                    continue
                seen_guids.add(visit_guid)
                if is_visit_processed(conn, visit_guid):
                    skipped_preprocessed += 1
                    log.info("Skipping already processed visit before evaluation index=%d visit_guid=%s", idx, visit_guid)
                    continue
                scheduled.append((idx, visit_guid, appointment))

            async def evaluate_and_persist() -> dict[str, int]:
                semaphore = asyncio.Semaphore(max(1, CONCURRENCY_N))

                async def run_job(
                    idx: int,
                    visit_guid: str,
                    appointment: dict[str, Any],
                ) -> tuple[int, str, MedKardRow | None, Exception | None]:
                    try:
                        row = await evaluate_with_limit(
                            semaphore=semaphore,
                            settings=settings,
                            appointment=appointment,
                            manifest_exact=manifest_exact,
                            manifest_group=manifest_group,
                        )
                        return idx, visit_guid, row, None
                    except Exception as exc:
                        return idx, visit_guid, None, exc

                tasks: list[asyncio.Task[tuple[int, str, MedKardRow | None, Exception | None]]] = []
                for idx, visit_guid, appointment in scheduled:
                    tasks.append(asyncio.create_task(run_job(idx, visit_guid, appointment)))

                stats = {"scheduled": len(tasks), "evaluated": 0, "failed": 0, "upserted": 0, "skipped_after_eval": 0}
                for task in asyncio.as_completed(tasks):
                    idx, initial_guid, row, error = await task
                    if error is not None:
                        stats["failed"] += 1
                        log.error(
                            "Skipping appointment due to evaluation error index=%d visit_guid=%s error=%s",
                            idx,
                            initial_guid or "<missing>",
                            error,
                        )
                        continue
                    if row is None:
                        stats["failed"] += 1
                        log.error(
                            "Skipping appointment due to empty evaluation result index=%d visit_guid=%s",
                            idx,
                            initial_guid or "<missing>",
                        )
                        continue
                    stats["evaluated"] += 1

                    if is_visit_processed(conn, row.visit_guid_1c):
                        stats["skipped_after_eval"] += 1
                        log.info(
                            "Skipping DB upsert after evaluation because visit already exists index=%d visit_guid=%s",
                            idx,
                            row.visit_guid_1c,
                        )
                        continue

                    upsert_medkard_rows(conn, [row])
                    conn.commit()
                    stats["upserted"] += 1
                    log.info("Persisted evaluated verdict index=%d visit_guid=%s", idx, row.visit_guid_1c)
                return stats

            stats = asyncio.run(evaluate_and_persist())
            verified = sum(1 for _, guid, _ in scheduled if is_visit_processed(conn, guid))

        print(
            json.dumps(
                {
                    "source": SOURCE_JSON_PATH,
                    "processed": len(appointments),
                    "scheduled": stats["scheduled"],
                    "evaluated": stats["evaluated"],
                    "failed": stats["failed"],
                    "upserted": stats["upserted"],
                    "skipped_missing_guid": skipped_missing_guid,
                    "skipped_preprocessed": skipped_preprocessed,
                    "skipped_after_eval": stats["skipped_after_eval"],
                    "verified_in_db": verified,
                    "db_write": verified > 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception:
        log.exception("File-to-DB verdict pipeline failed")
        raise


if __name__ == "__main__":
    main()
