from __future__ import annotations

from app.schemas.visit import CanonicalVisit
from app.services.heuristic_flags import VisitHeuristicFlagger


def test_heuristic_flags_detect_missing_and_placeholder() -> None:
    visit = CanonicalVisit(
        patient={},
        subjective={"complaints": "-"},
        objective={},
        assessment={},
        plan={},
    )

    flags = VisitHeuristicFlagger().generate_flags(visit)
    codes = {item.code for item in flags}

    assert "missing_required_field" in codes
    assert "placeholder_value" in codes
