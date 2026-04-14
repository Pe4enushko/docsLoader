from __future__ import annotations

"""High-level programmatic entry points for app initialization and pipelines."""

import asyncio
from pathlib import Path
from typing import Any

from sqlalchemy import delete

from app.config import get_settings
from app.models.db import SessionLocal, get_db_session
from app.models.knowledge import GuidelineDocument
from app.pipelines.batch_runner import AsyncVisitBatchRunner
from app.pipelines.guideline_ingestion import GuidelineIngestionPipeline, IngestionResult
from app.pipelines.visit_audit import VisitAuditPipeline, VisitAuditResult
from app.rag.postgres_adapter import PostgresRetrievalAdapter
from app.logger import configure_logging, get_logger
from app.utils.migrate import run_migrations


configure_logging()
log = get_logger(__name__)


def init_db() -> None:
    """Apply SQL migrations from `sql/migrations`."""
    run_migrations()


def ingest_guideline(source_file: str | Path) -> IngestionResult:
    """Ingest one clinical guideline document into normalized DB structures."""
    with get_db_session() as session:
        pipeline = GuidelineIngestionPipeline(session=session)
        return pipeline.ingest_file(source_file)


def ingest_all_guidelines_from_env() -> list[IngestionResult]:
    """Clear previous guideline ingestion and ingest all PDFs from env-defined directory."""
    settings = get_settings()
    base_dir = Path(settings.guidelines_dir)
    pattern = settings.guidelines_glob
    log.info(
        "Batch guideline ingestion started | dir=%s | pattern=%s | recursive=%s | clear_previous=%s",
        base_dir,
        pattern,
        settings.guidelines_recursive,
        settings.ingest_clear_previous,
    )

    if not base_dir.exists():
        raise FileNotFoundError(f"GUIDELINES_DIR not found: {base_dir}")

    files = sorted(base_dir.rglob(pattern) if settings.guidelines_recursive else base_dir.glob(pattern))
    files = [file for file in files if file.is_file()]
    log.info("Batch guideline ingestion files resolved | count=%s", len(files))

    results: list[IngestionResult] = []
    with SessionLocal() as session:
        if settings.ingest_clear_previous:
            delete_result = session.execute(delete(GuidelineDocument))
            session.commit()
            log.info(
                "Batch guideline ingestion cleanup done | deleted_rows=%s",
                getattr(delete_result, "rowcount", None),
            )

        pipeline = GuidelineIngestionPipeline(session=session)
        for idx, file in enumerate(files, start=1):
            log.info("Batch guideline ingestion file started | index=%s/%s | file=%s", idx, len(files), file)
            result = pipeline.ingest_file(file)
            session.commit()
            results.append(result)
            log.info(
                "Batch guideline ingestion file completed | index=%s/%s | file=%s | doc_id=%s",
                idx,
                len(files),
                file,
                result.document_id,
            )

    log.info("Batch guideline ingestion finished | files=%s", len(results))

    return results


def audit_visit(raw_visit: dict[str, Any], external_id: str | None = None) -> VisitAuditResult:
    """Audit one visit payload and persist generated report artifacts."""
    with get_db_session() as session:
        retrieval = PostgresRetrievalAdapter(session)
        pipeline = VisitAuditPipeline(session=session, retrieval_adapter=retrieval)
        return pipeline.process_one(raw_visit=raw_visit, external_id=external_id)


async def audit_visits_batch_async(
    visits: list[dict[str, Any]],
    *,
    max_concurrency: int = 4,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Audit a batch asynchronously with isolated sessions per task."""
    runner = AsyncVisitBatchRunner(session_factory=SessionLocal)
    return await runner.process_batch_async(
        visits=visits,
        max_concurrency=max_concurrency,
        continue_on_error=continue_on_error,
    )


def audit_visits_batch(
    visits: list[dict[str, Any]],
    *,
    max_concurrency: int = 4,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Synchronous wrapper over async batch API."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            audit_visits_batch_async(
                visits=visits,
                max_concurrency=max_concurrency,
                continue_on_error=continue_on_error,
            )
        )
    raise RuntimeError("audit_visits_batch cannot run inside active event loop. Use audit_visits_batch_async instead.")
