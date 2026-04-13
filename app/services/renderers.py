from __future__ import annotations

from app.schemas.visit import CanonicalVisit


class VisitRenderer:
    def to_markdown(self, visit: CanonicalVisit) -> str:
        blocks = [
            ("Meta", visit.meta),
            ("Patient", visit.patient),
            ("Subjective", visit.subjective),
            ("Objective", visit.objective),
            ("Assessment", visit.assessment),
            ("Plan", visit.plan),
            ("Admin", visit.admin),
        ]

        lines = ["# Normalized Visit Card", ""]
        for title, payload in blocks:
            lines.append(f"## {title}")
            if not payload:
                lines.append("- (empty)")
                lines.append("")
                continue
            for key, value in payload.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"
