from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.enums import SectionItemType, SectionType
from app.schemas.knowledge import NormalizedGuidelineDocument, NormalizedSection, NormalizedSectionItem, TitlePageMetadata
from app.utils.text import normalize_space, stable_hash


HEADING_PATTERNS: list[tuple[re.Pattern[str], SectionType]] = [
    (re.compile(r"^\s*оглавлен", re.IGNORECASE), SectionType.TOC),
    (re.compile(r"^\s*список\s+сокращ", re.IGNORECASE), SectionType.ABBREVIATIONS),
    (re.compile(r"^\s*термин", re.IGNORECASE), SectionType.DEFINITIONS),
    (re.compile(r"^\s*(1|i)\.?\s*кратк", re.IGNORECASE), SectionType.BRIEF_INFO),
    (re.compile(r"^\s*(2|ii)\.?\s*диагност", re.IGNORECASE), SectionType.DIAGNOSTICS),
    (re.compile(r"^\s*(3|iii)\.?\s*лечени", re.IGNORECASE), SectionType.TREATMENT),
    (re.compile(r"^\s*(4|iv)\.?\s*реабилитац", re.IGNORECASE), SectionType.REHABILITATION),
    (re.compile(r"^\s*(5|v)\.?\s*профилактик", re.IGNORECASE), SectionType.PREVENTION_FOLLOWUP),
    (re.compile(r"^\s*(6|vi)\.?\s*организац", re.IGNORECASE), SectionType.ORGANIZATION_OF_CARE),
    (re.compile(r"^\s*(7|vii)\.?\s*дополнитель", re.IGNORECASE), SectionType.ADDITIONAL_INFO),
    (re.compile(r"^\s*критерии\s+оценки\s+качества", re.IGNORECASE), SectionType.QUALITY_CRITERIA),
    (re.compile(r"^\s*список\s+литературы", re.IGNORECASE), SectionType.BIBLIOGRAPHY),
    (re.compile(r"^\s*приложен", re.IGNORECASE), SectionType.APPENDIX),
]


@dataclass(slots=True)
class RawSection:
    title: str
    body_lines: list[str]
    level: int
    order_index: int


def _looks_like_heading(line: str) -> bool:
    if not line:
        return False
    if re.match(r"^\d+(\.\d+)*\.?\s+", line):
        return True
    for pattern, _ in HEADING_PATTERNS:
        if pattern.search(line):
            return True
    return False


def _heading_level(title: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)", title)
    if not match:
        return 1
    return match.group(1).count(".") + 1


def _classify_section(title: str) -> SectionType:
    for pattern, section_type in HEADING_PATTERNS:
        if pattern.search(title):
            return section_type
    return SectionType.UNKNOWN


def _detect_item_type(text: str, section_type: SectionType) -> SectionItemType:
    lowered = text.lower()
    if "таблиц" in lowered:
        return SectionItemType.TABLE
    if "алгоритм" in lowered or "схем" in lowered:
        return SectionItemType.ALGORITHM
    if section_type == SectionType.APPENDIX:
        return SectionItemType.APPENDIX
    if any(marker in lowered for marker in ("рекомендуется", "следует", "показан", "необходимо")):
        return SectionItemType.RECOMMENDATION
    return SectionItemType.RAW_TEXT


class GuidelineSectionNormalizer:
    def normalize(self, source_path: str, extracted_text: str) -> NormalizedGuidelineDocument:
        lines = [line.strip() for line in extracted_text.splitlines()]
        lines = [line for line in lines if line]

        title_page = self._extract_title_page(lines)
        sections = self._split_sections(lines)

        normalized_sections = [
            self._to_normalized_section(raw_section)
            for raw_section in sections
        ]

        checksum = stable_hash(f"{source_path}:{extracted_text[:5000]}")
        return NormalizedGuidelineDocument(
            source_path=source_path,
            checksum=checksum,
            title_page=title_page,
            metadata={"line_count": len(lines)},
            sections=self._build_hierarchy(normalized_sections),
        )

    def _extract_title_page(self, lines: list[str]) -> TitlePageMetadata:
        head = lines[:60]
        title = next((line for line in head if len(line) > 12), "Клиническая рекомендация")

        icd_codes: list[str] = []
        for line in head:
            icd_codes.extend(re.findall(r"\b[A-ZА-Я]\d{2}(?:\.\d+)?\b", line.upper()))

        year = None
        for line in head:
            year_match = re.search(r"(20\d{2})", line)
            if year_match:
                year = int(year_match.group(1))
                break

        age_group = None
        for line in head:
            if any(token in line.lower() for token in ("взросл", "дет", "детск", "подрост")):
                age_group = normalize_space(line)
                break

        developer = None
        for line in head:
            lowered = line.lower()
            if "разработ" in lowered or "ассоциац" in lowered:
                developer = normalize_space(line)
                break

        return TitlePageMetadata(
            title=normalize_space(title),
            icd10_codes=sorted(set(icd_codes)),
            publication_year=year,
            age_group=age_group,
            developer=developer,
        )

    def _split_sections(self, lines: list[str]) -> list[RawSection]:
        sections: list[RawSection] = []
        current: RawSection | None = None

        for line in lines:
            if _looks_like_heading(line):
                if current:
                    sections.append(current)
                current = RawSection(
                    title=line,
                    body_lines=[],
                    level=_heading_level(line),
                    order_index=len(sections),
                )
                continue

            if current is None:
                current = RawSection(
                    title="Title Page",
                    body_lines=[],
                    level=1,
                    order_index=0,
                )
            current.body_lines.append(line)

        if current:
            sections.append(current)

        return sections

    def _to_normalized_section(self, section: RawSection) -> NormalizedSection:
        body = "\n".join(section.body_lines).strip()
        section_type = _classify_section(section.title)
        item = NormalizedSectionItem(
            item_type=_detect_item_type(body or section.title, section_type),
            text=body or section.title,
            metadata={"source_title": section.title},
        )

        return NormalizedSection(
            section_type=section_type,
            section_title=normalize_space(section.title),
            level=section.level,
            order_index=section.order_index,
            raw_text=body,
            cleaned_text=normalize_space(body),
            items=[item],
        )

    def _build_hierarchy(self, sections: list[NormalizedSection]) -> list[NormalizedSection]:
        if not sections:
            return []

        roots: list[NormalizedSection] = []
        stack: list[NormalizedSection] = []

        for section in sections:
            while stack and stack[-1].level >= section.level:
                stack.pop()

            if stack:
                stack[-1].subsections.append(section)
            else:
                roots.append(section)

            stack.append(section)

        return roots
