from __future__ import annotations

import re

from app.domain.enums import VisitType
from app.schemas.visit import CanonicalVisit, VisitClassificationResult


class VisitTypeClassifier:
    def classify(self, visit: CanonicalVisit) -> VisitClassificationResult:
        text_blobs = {
            "admin": " ".join(str(v) for v in visit.admin.values()).lower(),
            "assessment": " ".join(str(v) for v in visit.assessment.values()).lower(),
            "plan": " ".join(str(v) for v in visit.plan.values()).lower(),
            "subjective": " ".join(str(v) for v in visit.subjective.values()).lower(),
        }
        merged = " ".join(text_blobs.values())

        if self._is_prophylactic(merged):
            visit_type = VisitType.PROPHYLACTIC
            confidence = 0.87
            reasons = ["prophylactic markers in service/diagnosis/plan"]
        elif self._is_repeat(merged):
            visit_type = VisitType.REPEAT
            confidence = 0.8
            reasons = ["dynamic/effectiveness/follow-up markers detected"]
        elif self._is_primary(merged):
            visit_type = VisitType.PRIMARY
            confidence = 0.78
            reasons = ["first-visit markers and full baseline structure expected"]
        else:
            visit_type = VisitType.UNKNOWN
            confidence = 0.45
            reasons = ["insufficient explicit markers"]

        return VisitClassificationResult(
            visit_type=visit_type,
            confidence=confidence,
            reasons=reasons,
            prompt_conditions=self._build_prompt_conditions(visit_type),
        )

    def _is_prophylactic(self, text: str) -> bool:
        return any(token in text for token in ("профил", "диспанс", "скрининг", "медосмотр", "справк", "допуск"))

    def _is_repeat(self, text: str) -> bool:
        return any(token in text for token in ("повтор", "динам", "контроль", "эффект", "коррекц", "наблюдени"))

    def _is_primary(self, text: str) -> bool:
        return bool(re.search(r"(первичн|впервые|первый\s+визит)", text))

    def _build_prompt_conditions(self, visit_type: VisitType) -> list[str]:
        if visit_type == VisitType.PRIMARY:
            return [
                "Ожидаются жалобы, анамнез, объективный статус, диагноз, тактика.",
                "Оценить полноту базового первичного осмотра и обоснованность назначений.",
            ]
        if visit_type == VisitType.REPEAT:
            return [
                "Ожидается динамика состояния и оценка эффекта терапии.",
                "Проверить корректность дальнейшей тактики и follow-up.",
            ]
        if visit_type == VisitType.PROPHYLACTIC:
            return [
                "Отсутствие жалоб допустимо, но должен быть профилактический контекст решения.",
                "Лечение может отсутствовать; важна корректность заключения/допуска.",
            ]
        return ["Тип визита не определен; применить нейтральные правила полноты документации."]
