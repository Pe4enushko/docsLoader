from __future__ import annotations

import json

from app.schemas.retrieval import RetrievalContext
from app.schemas.visit import CanonicalVisit


def render_visit_compact(visit: CanonicalVisit) -> str:
    payload = {
        "subjective": visit.subjective,
        "objective": visit.objective,
        "assessment": visit.assessment,
        "plan": visit.plan,
        "admin": visit.admin,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_conditions(conditions: list[str]) -> str:
    if not conditions:
        return "- none"
    return "\n".join(f"- {item}" for item in conditions)


def render_retrieval_context(context: RetrievalContext) -> str:
    lines = ["### Retrieved Context", context.short_merged_context or "(empty)", "", "### References"]
    lines.extend(f"- {item}" for item in context.references_metadata[:20])
    return "\n".join(lines)
