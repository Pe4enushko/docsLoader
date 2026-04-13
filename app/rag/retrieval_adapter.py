from __future__ import annotations

"""Retrieval abstraction boundary for audit pipelines.

Audit logic must depend only on this interface, not on vector DB/SQL details.
"""

from abc import ABC, abstractmethod

from app.schemas.retrieval import RetrievalContext, RetrievalQuery


class RetrievalAdapter(ABC):
    """Contract for resolving structured context by diagnostic/query attributes."""

    @abstractmethod
    def retrieve_context(self, query: RetrievalQuery) -> RetrievalContext:
        """Return ready-to-use merged context for one audit stage."""
        raise NotImplementedError
