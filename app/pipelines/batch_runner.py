from __future__ import annotations

"""Helpers for processing batches of visits.

Contains:
- a simple synchronous runner for already constructed pipeline,
- an asynchronous runner that creates isolated DB sessions per task.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.integrations.one_c.parser import extract_visit_guid
from app.pipelines.visit_audit import VisitAuditPipeline, VisitAuditResult
from app.rag.postgres_adapter import PostgresRetrievalAdapter
from app.rag.retrieval_adapter import RetrievalAdapter
from app.logger import get_pipeline_logger


log = get_pipeline_logger(__name__, "visit_batch_runner.log")


def _resolve_external_id(visit: dict[str, Any]) -> str | None:
    """Resolve stable external id from generic or 1C appointment payload."""
    candidates = (
        visit.get("id"),
        visit.get("guid"),
        visit.get("GUID"),
        extract_visit_guid(visit),
    )
    for value in candidates:
        if value is None:
            continue
        external_id = str(value).strip()
        if external_id:
            return external_id
    return None


class VisitBatchRunner:
    """Runs one pre-built visit audit pipeline sequentially."""

    def __init__(self, audit_pipeline: VisitAuditPipeline) -> None:
        self.audit_pipeline = audit_pipeline

    def process_batch(self, visits: list[dict[str, Any]]) -> list[VisitAuditResult]:
        """Process all visits sequentially and return per-item outcomes."""
        results: list[VisitAuditResult] = []
        for item in visits:
            external_id = _resolve_external_id(item)
            results.append(self.audit_pipeline.process_one(raw_visit=item, external_id=external_id))
        return results


class AsyncVisitBatchRunner:
    """Runs visit audit in parallel with isolated sessions (safe for DB/ORM)."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        retrieval_factory: Callable[[Session], RetrievalAdapter] | None = None,
        pipeline_factory: Callable[[Session, RetrievalAdapter], VisitAuditPipeline] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.retrieval_factory = retrieval_factory or (lambda session: PostgresRetrievalAdapter(session))
        self.pipeline_factory = pipeline_factory or (
            lambda session, retrieval: VisitAuditPipeline(session=session, retrieval_adapter=retrieval)
        )

    async def process_batch_async(
        self,
        visits: list[dict[str, Any]],
        *,
        max_concurrency: int = 4,
        continue_on_error: bool = True,
    ) -> list[dict[str, Any]]:
        """Process visits concurrently and return normalized result payloads.

        Each item contains:
        - `status`: `ok` or `error`
        - ids/status on success
        - error message on failure
        """
        concurrency = max(1, int(max_concurrency))
        semaphore = asyncio.Semaphore(concurrency)
        stop_requested = False
        results: list[dict[str, Any] | None] = [None] * len(visits)

        async def run_one(index: int, visit: dict[str, Any]) -> None:
            nonlocal stop_requested
            if stop_requested:
                return

            external_id = _resolve_external_id(visit)
            async with semaphore:
                if stop_requested:
                    return
                try:
                    result = await asyncio.to_thread(self._process_one_sync, visit, external_id)
                    results[index] = {
                        "index": index + 1,
                        "external_id": external_id,
                        "status": "ok",
                        "visit_id": str(result.visit_id),
                        "report_id": str(result.report_id),
                        "pipeline_status": result.status,
                    }
                    log.info(
                        "Async batch item completed | index=%s/%s | external_id=%s | visit_id=%s | report_id=%s",
                        index + 1,
                        len(visits),
                        external_id,
                        result.visit_id,
                        result.report_id,
                    )
                except Exception as exc:
                    results[index] = {
                        "index": index + 1,
                        "external_id": external_id,
                        "status": "error",
                        "error": str(exc),
                    }
                    log.exception(
                        "Async batch item failed | index=%s/%s | external_id=%s",
                        index + 1,
                        len(visits),
                        external_id,
                    )
                    if not continue_on_error:
                        stop_requested = True

        tasks = [asyncio.create_task(run_one(index, visit)) for index, visit in enumerate(visits)]
        await asyncio.gather(*tasks)
        return [item for item in results if item is not None]

    def _process_one_sync(self, visit: dict[str, Any], external_id: str | None) -> VisitAuditResult:
        """Run one visit in dedicated session (used by async runner thread)."""
        session = self.session_factory()
        try:
            retrieval = self.retrieval_factory(session)
            pipeline = self.pipeline_factory(session, retrieval)
            result = pipeline.process_one(raw_visit=visit, external_id=external_id)
            session.commit()
            return result
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
