from __future__ import annotations

"""Section-level write operations for normative documents."""

from uuid import UUID

from sqlalchemy.orm import Session

from app.models.knowledge import NormativeSection


class NormativeSectionStorage:
    """Stores normalized section rows for normative documents."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_section(
        self,
        *,
        document_id: UUID,
        section_title: str,
        raw_text: str,
        cleaned_text: str,
        order_index: int,
        level: int = 1,
    ) -> NormativeSection:
        section = NormativeSection(
            document_id=document_id,
            section_title=section_title,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            order_index=order_index,
            level=level,
        )
        self.session.add(section)
        self.session.flush()
        return section
