from __future__ import annotations

from app.domain.enums import SectionType
from app.ingestion.section_normalizer import GuidelineSectionNormalizer


def _flatten(sections):
    for section in sections:
        yield section
        yield from _flatten(section.subsections)


def test_guideline_normalizer_maps_typical_sections() -> None:
    text = """
Клиническая рекомендация по заболеванию
МКБ A09
2024

1. Краткая информация
Определение заболевания.

2. Диагностика
Рекомендуется оценить жалобы и анамнез.

3. Лечение
Рекомендуется симптоматическая терапия.

Приложение А1
Алгоритм ведения пациента.
""".strip()

    normalizer = GuidelineSectionNormalizer()
    doc = normalizer.normalize(source_path="sample.pdf", extracted_text=text)
    all_sections = list(_flatten(doc.sections))
    section_types = {section.section_type for section in all_sections}

    assert doc.title_page.title
    assert "A09" in doc.title_page.icd10_codes
    assert SectionType.BRIEF_INFO in section_types
    assert SectionType.DIAGNOSTICS in section_types
    assert SectionType.TREATMENT in section_types
    assert SectionType.APPENDIX in section_types
