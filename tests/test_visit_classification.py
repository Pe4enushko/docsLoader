from __future__ import annotations

from app.domain.enums import VisitType
from app.schemas.visit import CanonicalVisit
from app.services.visit_classifier import VisitTypeClassifier


def test_visit_classifier_primary() -> None:
    visit = CanonicalVisit(
        admin={"service_text": "первичный прием терапевта"},
        subjective={"complaints": "кашель"},
        objective={"status": "умеренное"},
        assessment={"diagnosis": "J20.9"},
        plan={"plan": "назначение терапии"},
    )
    result = VisitTypeClassifier().classify(visit)
    assert result.visit_type == VisitType.PRIMARY


def test_visit_classifier_repeat() -> None:
    visit = CanonicalVisit(
        admin={"service_text": "повторный прием"},
        subjective={"complaints": "лучше"},
        objective={"status": "динамика положительная"},
        assessment={"diagnosis": "J20.9"},
        plan={"plan": "контроль через 5 дней"},
    )
    result = VisitTypeClassifier().classify(visit)
    assert result.visit_type == VisitType.REPEAT


def test_visit_classifier_prophylactic() -> None:
    visit = CanonicalVisit(
        admin={"service_text": "профилактический медосмотр"},
        assessment={"diagnosis": "Z00.0"},
        plan={"plan": "допуск"},
    )
    result = VisitTypeClassifier().classify(visit)
    assert result.visit_type == VisitType.PROPHYLACTIC
