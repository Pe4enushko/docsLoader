from __future__ import annotations

"""Write operations for async processing jobs queue."""

from uuid import UUID

from sqlalchemy.orm import Session

from app.models.visit import ProcessingJob


class ProcessingJobStorage:
    """Stores queue status transitions for processing jobs."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def enqueue(self, job_type: str, entity_id: UUID, payload: dict | None = None) -> ProcessingJob:
        row = ProcessingJob(job_type=job_type, entity_id=entity_id, status="queued", payload=payload or {})
        self.session.add(row)
        self.session.flush()
        return row

    def set_status(self, job: ProcessingJob, *, status: str, error_message: str | None = None) -> ProcessingJob:
        job.status = status
        job.error_message = error_message
        self.session.flush()
        return job
