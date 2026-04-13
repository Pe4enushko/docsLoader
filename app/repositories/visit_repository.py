from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.enums import VisitType
from app.models.visit import VisitRecord


class VisitRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_raw_visit(self, raw_json: dict[str, Any], external_id: str | None = None) -> VisitRecord:
        visit = VisitRecord(raw_json=raw_json, external_id=external_id)
        self.session.add(visit)
        self.session.flush()
        return visit

    def get_by_id(self, visit_id: UUID) -> VisitRecord | None:
        stmt = select(VisitRecord).where(VisitRecord.id == visit_id)
        return self.session.scalar(stmt)

    def update_preprocessed(
        self,
        visit: VisitRecord,
        normalized_json: dict[str, Any],
        readable_render: str,
        visit_type: VisitType,
        specialty: str | None,
        patient_age: int | None,
        icd10_codes: list[str],
        extraction_flags: list[str],
        metadata_json: dict[str, Any],
    ) -> VisitRecord:
        visit.normalized_json = normalized_json
        visit.readable_render = readable_render
        visit.visit_type = visit_type
        visit.specialty = specialty
        visit.patient_age = patient_age
        visit.icd10_codes = icd10_codes
        visit.extraction_flags = extraction_flags
        visit.metadata_json = metadata_json
        self.session.flush()
        return visit
