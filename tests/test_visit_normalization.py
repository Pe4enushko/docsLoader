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


def test_visit_normalizer_supports_one_c_appointment_item() -> None:
    payload = {
        "Прием": {"GUID": "6957b3d6-0795-11f1-a221-00155daa6107", "DATE": "12.02.2026"},
        "Врач": {"SPECIALIZATION": "Педиатр"},
        "Пациент": {"GENDER": "Мужской", "AGE": "4"},
        "Услуги": [
            {
                "КодЕГИСЗ": "B01.031.002",
                "Наименование": "Прием (осмотр, консультация) врача-педиатра повторный",
            }
        ],
        "ДанныеОсмотра": [
            {"Параметр": "Жалобы на момент осмотра", "Значение": "Кашель"},
            {"Параметр": "Температура", "Значение": "36,6"},
            {"Параметр": "План лечения", "Значение": "Симптоматическая терапия"},
        ],
        "Диагнозы": [
            {
                "КодМКБ": "J06.9",
                "НаименованиеМКБ": "Острая инфекция верхних дыхательных путей неуточненная",
                "Детализация": "Острый ринофаринготрахеит",
                "ВыявленВпервые": False,
            }
        ],
    }

    visit = VisitNormalizer().normalize(payload)

    assert visit.meta.get("visit_guid") == "6957b3d6-0795-11f1-a221-00155daa6107"
    assert visit.patient.get("age") == "4"
    assert visit.admin.get("specialty") == "Педиатр"
    assert "повторный" in str(visit.admin.get("service_text", "")).lower()
    assert "кашель" in str(visit.subjective.values()).lower()
    assert "температура" in " ".join(visit.objective.keys()).lower()
    assert "план_лечения" in visit.plan
    assert visit.assessment.get("icd10_codes") == ["J06.9"]
