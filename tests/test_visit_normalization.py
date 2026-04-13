from __future__ import annotations

from app.services.visit_normalizer import VisitNormalizer


def test_visit_normalizer_builds_canonical_structure() -> None:
    payload = {
        "Пациент": {"Возраст": 45, "Пол": "M"},
        "Жалобы": {"text": "кашель"},
        "Осмотр": {"temp": "37.2"},
        "Диагноз": {"main": "J06.9 ОРВИ"},
        "Рекомендации": {"plan": "обильное питье"},
        "Услуга": {"specialty": "therapy"},
    }

    visit = VisitNormalizer().normalize(payload)

    assert visit.patient
    assert visit.subjective
    assert visit.objective
    assert visit.assessment
    assert visit.plan
    assert visit.admin
