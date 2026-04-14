from __future__ import annotations

"""Prompt-injection provider for normative JSON rules.

The provider is intentionally separated from RAG so normative constraints can be
maintained as explicit policy rules and injected only into selected stages.
"""

import json
from pathlib import Path

from app.config import get_settings
from app.domain.enums import VisitType
from app.schemas.normative_prompt_rules import NormativePromptRule
from app.utils.logging import get_logger


log = get_logger(__name__)


class NormativePromptRuleProvider:
    """Loads and filters normative rules for prompt injection."""

    def __init__(self, rules_path: str | Path | None = None) -> None:
        settings = get_settings()
        self.rules_path = self._resolve_rules_path(Path(rules_path or settings.normative_rules_json_path))
        self._cache_mtime_ns: int | None = None
        self._cache_rules: list[NormativePromptRule] = []

    def get_applicable_rules(
        self,
        *,
        visit_type: VisitType,
        specialty: str | None,
        patient_age: int | None,
    ) -> list[NormativePromptRule]:
        """Return rules matching visit type, specialty and age group."""
        rules = self._load_rules()
        specialty_token = self._normalize_specialty(specialty)
        age_group = self._resolve_age_group(patient_age)

        applicable: list[NormativePromptRule] = []
        for rule in rules:
            applies = rule.applies_to
            if applies.visit_types and visit_type not in applies.visit_types:
                continue
            if applies.specialties and not self._specialty_matches(applies.specialties, specialty_token):
                continue
            if applies.age_group and not self._age_group_matches(applies.age_group, age_group):
                continue
            applicable.append(rule)
        return applicable

    def render_for_prompt(self, rules: list[NormativePromptRule]) -> list[str]:
        """Render filtered rules as compact lines suitable for prompt conditions."""
        rendered: list[str] = []
        for rule in rules:
            targets = ", ".join(rule.targets) if rule.targets else "any_field"
            line = (
                f"[{rule.flag_code}] ({rule.severity}, src={rule.source}, type={rule.rule_type}) "
                f"Targets: {targets}. Expectation: {rule.expectation}"
            )
            if rule.condition:
                line += f" Condition: {rule.condition}"
            rendered.append(line)
        return rendered

    def to_reference_metadata(self, rules: list[NormativePromptRule]) -> list[dict[str, str]]:
        """Build lightweight references for report traceability."""
        return [
            {
                "type": "normative_prompt_rule",
                "rule_id": rule.rule_id,
                "source": rule.source,
                "flag_code": rule.flag_code,
                "severity": rule.severity,
            }
            for rule in rules
        ]

    def _load_rules(self) -> list[NormativePromptRule]:
        """Lazy-load rules from disk with mtime-based reload."""
        if not self.rules_path.exists():
            log.warning("Normative rules JSON not found | path=%s", self.rules_path)
            return []

        stat = self.rules_path.stat()
        if self._cache_mtime_ns == stat.st_mtime_ns and self._cache_rules:
            return self._cache_rules

        payload = json.loads(self.rules_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Normative rules JSON must be array: {self.rules_path}")

        rules = [NormativePromptRule.model_validate(item) for item in payload if isinstance(item, dict)]
        self._cache_rules = rules
        self._cache_mtime_ns = stat.st_mtime_ns
        log.info("Loaded normative prompt rules | path=%s | rules=%s", self.rules_path, len(rules))
        return rules

    def _resolve_rules_path(self, path: Path) -> Path:
        """Resolve relative rules path against project root.

        This keeps configuration stable when scripts are started from another cwd.
        """
        if path.is_absolute():
            return path
        project_root = Path(__file__).resolve().parents[2]
        return (project_root / path).resolve()

    def _normalize_specialty(self, specialty: str | None) -> str | None:
        if not specialty:
            return None
        value = specialty.strip().lower().replace("ё", "е")
        if "педиатр" in value or "pediatric" in value:
            return "pediatrics"
        return value

    def _specialty_matches(self, allowed: list[str], actual: str | None) -> bool:
        if actual is None:
            return False
        normalized_allowed = {item.strip().lower().replace("ё", "е") for item in allowed}
        return actual in normalized_allowed

    def _resolve_age_group(self, age: int | None) -> str | None:
        if age is None:
            return None
        return "child" if age < 18 else "adult"

    def _age_group_matches(self, expected: str, actual: str | None) -> bool:
        return expected.strip().lower() == (actual or "").strip().lower()
