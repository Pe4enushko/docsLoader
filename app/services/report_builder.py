from __future__ import annotations

from app.schemas.llm import StageCheckResult
from app.schemas.report import AuditReportPayload, SectionFinding
from app.schemas.visit import HeuristicFlag, VisitClassificationResult


class AuditReportBuilder:
    def build(
        self,
        classification: VisitClassificationResult,
        heuristic_flags: list[HeuristicFlag],
        stage_results: list[StageCheckResult],
        references: list[dict],
    ) -> AuditReportPayload:
        structural = [flag.message for flag in heuristic_flags if flag.code in {"missing_required_field", "duplicate_parameter"}]
        logical = [flag.message for flag in heuristic_flags if flag.code not in {"missing_required_field", "duplicate_parameter"}]

        diagnostic_flags = []
        management_flags = []
        section_findings: list[SectionFinding] = []
        human_review_required = False

        for result in stage_results:
            stage_messages = [str(item.get("message", "")) for item in result.findings]
            section_findings.append(SectionFinding(section=result.stage.value, findings=stage_messages or [result.summary]))

            if "diagnosis" in result.stage.value:
                diagnostic_flags.extend(stage_messages)
            if "management" in result.stage.value:
                management_flags.extend(stage_messages)
            if result.status.lower() not in {"ok", "pass"}:
                human_review_required = True

        if any(flag.severity == "error" for flag in heuristic_flags):
            human_review_required = True

        followup_adequacy = "adequate"
        if any("follow" in finding.lower() or "наблюд" in finding.lower() for finding in logical):
            followup_adequacy = "needs_attention"

        final_summary = self._build_summary(classification, structural, logical, human_review_required)

        return AuditReportPayload(
            structural_issues=structural,
            logical_issues=logical,
            diagnostic_flags=diagnostic_flags,
            management_flags=management_flags,
            followup_adequacy=followup_adequacy,
            human_review_required=human_review_required,
            section_wise_findings=section_findings,
            final_summary=final_summary,
            stage_results=stage_results,
            heuristic_flags=heuristic_flags,
            classification=classification,
            references=references,
            scores={
                "heuristic_issue_count": len(heuristic_flags),
                "stage_issue_count": sum(len(stage.findings) for stage in stage_results),
            },
        )

    def to_text(self, report: AuditReportPayload) -> str:
        lines = [
            "# Audit Report",
            "",
            f"Visit type: {report.classification.visit_type.value} (confidence={report.classification.confidence:.2f})",
            f"Human review required: {'yes' if report.human_review_required else 'no'}",
            "",
            "## Structural issues",
        ]
        lines.extend(f"- {item}" for item in report.structural_issues or ["none"])
        lines.extend(["", "## Logical issues"])
        lines.extend(f"- {item}" for item in report.logical_issues or ["none"])
        lines.extend(["", "## Final summary", report.final_summary, ""])
        return "\n".join(lines)

    def _build_summary(
        self,
        classification: VisitClassificationResult,
        structural_issues: list[str],
        logical_issues: list[str],
        human_review_required: bool,
    ) -> str:
        if human_review_required:
            return (
                f"Запись классифицирована как {classification.visit_type.value}. "
                f"Обнаружены структурные ({len(structural_issues)}) и логические ({len(logical_issues)}) замечания; "
                "рекомендуется ручная проверка врачом-экспертом."
            )
        return (
            f"Запись классифицирована как {classification.visit_type.value}. "
            "Критичных несоответствий не обнаружено, возможна автоматическая приемка с выборочным контролем."
        )
