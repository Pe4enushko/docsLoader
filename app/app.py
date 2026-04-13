from __future__ import annotations

"""High-level programmatic entry points for app initialization and pipelines."""

from pathlib import Path
from typing import Any

from sqlalchemy import delete

from app.config import get_settings
from app.models import Base
from app.models.db import SessionLocal, engine, get_db_session
from app.models.knowledge import GuidelineDocument
from app.pipelines.guideline_ingestion import GuidelineIngestionPipeline, IngestionResult
from app.pipelines.visit_audit import VisitAuditPipeline, VisitAuditResult
from app.rag.postgres_adapter import PostgresRetrievalAdapter
from app.utils.logging import get_logger


log = get_logger(__name__)


def init_db() -> None:
    """Create all tables from ORM metadata (useful for local bootstrapping)."""
    Base.metadata.create_all(bind=engine)


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
