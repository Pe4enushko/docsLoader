from __future__ import annotations

from sqlalchemy import Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import TypeEngine

try:
    from pgvector.sqlalchemy import Vector
except Exception:  # pragma: no cover
    Vector = None


def vector_type(dimensions: int) -> TypeEngine:
    if Vector is not None:
        return Vector(dimensions)
    return ARRAY(Float)
