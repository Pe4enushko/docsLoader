from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.Storage import VisitRecordStorage
from app.domain.enums import VisitType
from app.models.visit import VisitRecord


class VisitRepository:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.visit_storage = VisitRecordStorage(session)

    def create_raw_visit(self, raw_json: dict[str, Any], external_id: str | None = None) -> VisitRecord:
        return self.visit_storage.create_raw_visit(raw_json=raw_json, external_id=external_id)

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
        return self.visit_storage.update_preprocessed(
            visit,
            normalized_json=normalized_json,
            readable_render=readable_render,
            visit_type=visit_type,
            specialty=specialty,
            patient_age=patient_age,
            icd10_codes=icd10_codes,
            extraction_flags=extraction_flags,
            metadata_json=metadata_json,
        )
