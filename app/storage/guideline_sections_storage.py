from __future__ import annotations

"""Section-level write operations for normalized guideline hierarchy."""

from collections.abc import Iterable

from sqlalchemy.orm import Session

from app.models.knowledge import GuidelineDocument, GuidelineSection
from app.schemas.knowledge import NormalizedSection


class GuidelineSectionStorage:
    """Stores recursive section/subsection tree for one guideline document."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def store_sections(
        self,
        document: GuidelineDocument,
        sections: Iterable[NormalizedSection],
        *,
        parent: GuidelineSection | None = None,
    ) -> None:
        for section in sections:
            record = GuidelineSection(
                document_id=document.id,
                parent_section_id=parent.id if parent else None,
                section_type=section.section_type,
                section_title=section.section_title,
                level=section.level,
                order_index=section.order_index,
                page_from=section.page_from,
                page_to=section.page_to,
                raw_text=section.raw_text,
                cleaned_text=section.cleaned_text,
                metadata_json={
                    "items": [item.model_dump(mode="json") for item in section.items],
                },
            )
            self.session.add(record)
            self.session.flush()
            self.store_sections(document, section.subsections, parent=record)
