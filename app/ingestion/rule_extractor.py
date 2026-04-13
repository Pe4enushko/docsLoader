from __future__ import annotations

"""LLM-assisted extraction of auditable rules from normalized guideline text.

This extractor first asks an LLM to convert narrative guideline fragments into
structured rule candidates. If the model output is invalid/unavailable, it
falls back to conservative heuristic extraction.
"""

import json
import re
from typing import Any

from app.config import get_settings
from app.domain.enums import RuleType
from app.schemas.knowledge import NormalizedGuidelineDocument, RuleCandidate
from app.services.llm_client import LLMClient, OpenAILLMClient
from app.utils.logging import get_logger
from app.utils.text import stable_hash


log = get_logger(__name__)

RECOMMENDATION_HINTS = (
    "рекомендуется",
    "следует",
    "необходимо",
    "показан",
    "показано",
    "должен",
    "should",
    "recommended",
)

RULE_TYPE_MAP = {
    "diagnostic": RuleType.DIAGNOSTIC,
    "management": RuleType.MANAGEMENT,
    "followup": RuleType.FOLLOWUP,
    "documentation": RuleType.DOCUMENTATION,
    "quality": RuleType.QUALITY,
    "other": RuleType.OTHER,
}


class GuidelineRuleExtractor:
    """Extract rules from normalized guideline sections.

    Strategy:
    1. Split large sections into LLM-friendly fragments.
    2. Ask LLM to return strict JSON with extracted rules.
    3. Validate and normalize records into `RuleCandidate`.
    4. Fallback to heuristic extraction if parsing fails.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        model: str | None = None,
        max_fragment_chars: int = 5000,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.rule_extractor_model
        self.max_fragment_chars = max_fragment_chars
        self.llm_client = llm_client or OpenAILLMClient(default_model=self.model)

    def extract(self, doc: NormalizedGuidelineDocument) -> list[RuleCandidate]:
        """Extract and deduplicate rule candidates from the entire document."""
        rules: list[RuleCandidate] = []
        seen: set[str] = set()

        for section in self._walk_sections(doc.sections):
            section_text = section.cleaned_text or section.raw_text
            if not section_text:
                continue

            for fragment in self._split_for_llm(section_text):
                # Primary path: ask model to return structured rules.
                llm_rules = self._extract_with_llm(
                    section_title=section.section_title,
                    section_type=section.section_type.value,
                    text_fragment=fragment,
                )

                # Safety net: if LLM returned nothing/invalid, use heuristics.
                current_rules = llm_rules or self._extract_with_heuristics(
                    section_title=section.section_title,
                    section_type=section.section_type.value,
                    text_fragment=fragment,
                )

                for candidate in current_rules:
                    key = stable_hash(f"{candidate.topic}:{candidate.statement}")
                    if key in seen:
                        continue
                    seen.add(key)
                    rules.append(candidate)

        log.info("Rule extraction finished | source=%s | rules=%s", doc.source_path, len(rules))
        return rules

    def _extract_with_llm(self, *, section_title: str, section_type: str, text_fragment: str) -> list[RuleCandidate]:
        """Call LLM and parse structured rule objects from JSON response."""
        system_prompt = (
            "You extract auditable medical guideline rules. "
            "Return ONLY valid JSON object with schema: "
            "{\"rules\":[{\"topic\":str,\"rule_type\":str,\"statement\":str,"
            "\"conditions\":list[str],\"triggers\":list[str],\"audit_targets\":list[str],"
            "\"population\":str|null,\"specialty\":str|null,\"source_quote\":str|null}]}. "
            "rule_type must be one of: diagnostic, management, followup, documentation, quality, other."
        )
        user_prompt = (
            "Extract only explicit actionable/checkable rules from the fragment below.\n"
            f"Section title: {section_title}\n"
            f"Section type: {section_type}\n"
            "Text:\n"
            f"{text_fragment}"
        )

        try:
            response = self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
            )
            payload = self._parse_json_payload(response.text)
            raw_rules = payload.get("rules", []) if isinstance(payload, dict) else []
            if not isinstance(raw_rules, list):
                return []

            parsed_rules: list[RuleCandidate] = []
            for item in raw_rules:
                if not isinstance(item, dict):
                    continue
                statement = str(item.get("statement", "")).strip()
                if len(statement) < 20:
                    continue

                rule_type_raw = str(item.get("rule_type", "other")).strip().lower()
                rule_type = RULE_TYPE_MAP.get(rule_type_raw, self._infer_rule_type(section_type, statement.lower()))

                parsed_rules.append(
                    RuleCandidate(
                        topic=str(item.get("topic", section_title)).strip() or section_title,
                        population=self._to_optional_str(item.get("population")),
                        specialty=self._to_optional_str(item.get("specialty")),
                        rule_type=rule_type,
                        statement=statement,
                        conditions=self._to_str_list(item.get("conditions")),
                        triggers=self._to_str_list(item.get("triggers")),
                        audit_targets=self._to_str_list(item.get("audit_targets")) or [section_type],
                        source_quote=self._to_optional_str(item.get("source_quote")) or statement,
                        source_section=section_title,
                        metadata={
                            "extraction_method": "llm",
                            "section_type": section_type,
                        },
                    )
                )
            return parsed_rules
        except Exception:
            log.exception("LLM-assisted rule extraction failed | section=%s", section_title)
            return []

    def _extract_with_heuristics(self, *, section_title: str, section_type: str, text_fragment: str) -> list[RuleCandidate]:
        """Heuristic fallback used when LLM response is unavailable/invalid."""
        rules: list[RuleCandidate] = []
        sentences = re.split(r"(?<=[.!?])\s+", text_fragment)

        for sentence in sentences:
            normalized_sentence = sentence.strip()
            if len(normalized_sentence) < 40:
                continue

            lowered = normalized_sentence.lower()
            if not any(hint in lowered for hint in RECOMMENDATION_HINTS):
                continue

            rules.append(
                RuleCandidate(
                    topic=section_title,
                    specialty=None,
                    rule_type=self._infer_rule_type(section_type, lowered),
                    statement=normalized_sentence,
                    conditions=self._extract_conditions(normalized_sentence),
                    triggers=self._extract_triggers(normalized_sentence),
                    audit_targets=[section_type],
                    source_quote=normalized_sentence,
                    source_section=section_title,
                    metadata={
                        "extraction_method": "heuristic",
                        "section_type": section_type,
                    },
                )
            )

        return rules

    def _walk_sections(self, sections):
        """Depth-first traversal over section tree."""
        for section in sections:
            yield section
            yield from self._walk_sections(section.subsections)

    def _split_for_llm(self, text: str) -> list[str]:
        """Split long text into bounded fragments suitable for model context windows."""
        payload = text.strip()
        if len(payload) <= self.max_fragment_chars:
            return [payload]

        parts: list[str] = []
        start = 0
        while start < len(payload):
            end = min(len(payload), start + self.max_fragment_chars)
            fragment = payload[start:end].strip()
            if fragment:
                parts.append(fragment)
            if end == len(payload):
                break
            start = end
        return parts

    def _parse_json_payload(self, text: str) -> Any:
        """Parse JSON from raw model output (plain, fenced or noisy)."""
        raw = text.strip()
        if not raw:
            return {}

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            try:
                return json.loads(fenced_match.group(1))
            except json.JSONDecodeError:
                pass

        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1 and last > first:
            return json.loads(raw[first : last + 1])

        raise ValueError("Cannot parse JSON from LLM output")

    def _infer_rule_type(self, section_type: str, text: str) -> RuleType:
        """Infer semantic rule type from section context and sentence wording."""
        if "диагност" in section_type or "анамнез" in text:
            return RuleType.DIAGNOSTIC
        if "лечени" in section_type or "терап" in text:
            return RuleType.MANAGEMENT
        if "наблюден" in text or "follow" in text:
            return RuleType.FOLLOWUP
        if "докум" in text or "запис" in text:
            return RuleType.DOCUMENTATION
        if "качест" in text or "критери" in text:
            return RuleType.QUALITY
        return RuleType.OTHER

    def _extract_conditions(self, statement: str) -> list[str]:
        """Extract coarse condition candidates from recommendation statement."""
        lower = statement.lower()
        for marker in ("при ", "если ", "в случае "):
            if marker in lower:
                return [statement.strip()]
        return []

    def _extract_triggers(self, statement: str) -> list[str]:
        """Extract short trigger phrases near condition markers."""
        triggers: list[str] = []
        tokens = statement.split()
        for idx, token in enumerate(tokens):
            if token.lower() in {"при", "если", "when", "if"}:
                triggers.append(" ".join(tokens[idx : idx + 6]))
        return triggers

    def _to_str_list(self, payload: Any) -> list[str]:
        """Normalize unknown payload to cleaned list[str]."""
        if payload is None:
            return []
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        if isinstance(payload, str):
            value = payload.strip()
            return [value] if value else []
        return []

    def _to_optional_str(self, payload: Any) -> str | None:
        """Normalize optional scalar values from model payload."""
        if payload is None:
            return None
        value = str(payload).strip()
        return value or None
