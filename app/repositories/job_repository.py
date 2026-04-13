from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.visit import ProcessingJob


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def enqueue(self, job_type: str, entity_id: str, payload: dict | None = None) -> ProcessingJob:
        row = ProcessingJob(job_type=job_type, entity_id=entity_id, status="queued", payload=payload or {})
        self.session.add(row)
        self.session.flush()
        return row

    def set_status(self, job: ProcessingJob, status: str, error_message: str | None = None) -> ProcessingJob:
        job.status = status
        job.error_message = error_message
        self.session.flush()
        return job

    def next_queued(self, job_type: str | None = None) -> ProcessingJob | None:
        stmt = select(ProcessingJob).where(ProcessingJob.status == "queued").order_by(ProcessingJob.created_at.asc())
        if job_type:
            stmt = stmt.where(ProcessingJob.job_type == job_type)
        return self.session.scalar(stmt)
