from __future__ import annotations

import json

from app.domain.enums import VisitType
from app.services.normative_prompt_rules import NormativePromptRuleProvider


def test_normative_prompt_rules_are_filtered_by_visit_type_and_population(tmp_path) -> None:
    payload = [
        {
            "rule_id": "all_visits",
            "source": "274n",
            "rule_type": "required_field",
            "applies_to": {"visit_types": ["primary", "repeat", "prophylactic"], "specialties": ["pediatrics"], "age_group": "child"},
            "targets": ["visit_date"],
            "expectation": "meta required",
            "flag_code": "META_MISSING",
            "severity": "critical",
        },
        {
            "rule_id": "only_repeat",
            "source": "203n",
            "rule_type": "required_field",
            "applies_to": {"visit_types": ["repeat"], "specialties": ["pediatrics"], "age_group": "child"},
            "targets": ["dynamics"],
            "expectation": "repeat requires dynamics",
            "flag_code": "REPEAT_DYNAMICS_MISSING",
            "severity": "major",
        },
    ]
    rules_file = tmp_path / "rules.json"
    rules_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    provider = NormativePromptRuleProvider(rules_path=rules_file)

    primary_rules = provider.get_applicable_rules(
        visit_type=VisitType.PRIMARY,
        specialty="Педиатр",
        patient_age=7,
    )
    repeat_rules = provider.get_applicable_rules(
        visit_type=VisitType.REPEAT,
        specialty="Педиатр",
        patient_age=7,
    )

    assert {item.rule_id for item in primary_rules} == {"all_visits"}
    assert {item.rule_id for item in repeat_rules} == {"all_visits", "only_repeat"}


def test_normative_prompt_rules_render_lines(tmp_path) -> None:
    payload = [
        {
            "rule_id": "sample",
            "source": "274n",
            "rule_type": "required_field",
            "applies_to": {"visit_types": ["primary"], "specialties": ["pediatrics"], "age_group": "child"},
            "targets": ["objective_exam"],
            "expectation": "objective exam required",
            "flag_code": "MISSING_OBJECTIVE_EXAM",
            "severity": "critical",
        }
    ]
    rules_file = tmp_path / "rules.json"
    rules_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    provider = NormativePromptRuleProvider(rules_path=rules_file)
    rules = provider.get_applicable_rules(visit_type=VisitType.PRIMARY, specialty="pediatrics", patient_age=9)
    lines = provider.render_for_prompt(rules)

    assert len(lines) == 1
    assert "MISSING_OBJECTIVE_EXAM" in lines[0]
    assert "objective exam required" in lines[0]
