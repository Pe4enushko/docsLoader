from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.Storage import ProcessingJobStorage
from app.models.visit import ProcessingJob


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.job_storage = ProcessingJobStorage(session)

    def enqueue(self, job_type: str, entity_id: UUID, payload: dict | None = None) -> ProcessingJob:
        return self.job_storage.enqueue(job_type=job_type, entity_id=entity_id, payload=payload)

    def set_status(self, job: ProcessingJob, status: str, error_message: str | None = None) -> ProcessingJob:
        return self.job_storage.set_status(job, status=status, error_message=error_message)

    def next_queued(self, job_type: str | None = None) -> ProcessingJob | None:
        stmt = select(ProcessingJob).where(ProcessingJob.status == "queued").order_by(ProcessingJob.created_at.asc())
        if job_type:
            stmt = stmt.where(ProcessingJob.job_type == job_type)
        return self.session.scalar(stmt)
