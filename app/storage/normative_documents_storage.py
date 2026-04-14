from __future__ import annotations

"""Document-level write operations for normative ingestion."""

from sqlalchemy.orm import Session

from app.models.knowledge import NormativeDocument


class NormativeDocumentStorage:
    """Stores normative document metadata rows."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_document(
        self,
        *,
        title: str,
        source_path: str,
        checksum: str,
        metadata: dict,
    ) -> NormativeDocument:
        doc = NormativeDocument(
            title=title,
            source_path=source_path,
            checksum=checksum,
            metadata_json=metadata,
        )
        self.session.add(doc)
        self.session.flush()
        return doc
