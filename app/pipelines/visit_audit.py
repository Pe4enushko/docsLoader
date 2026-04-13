from __future__ import annotations

"""Pipeline for auditing one medical visit record.

The flow normalizes raw visit JSON, generates heuristic flags, classifies visit
type, retrieves context via adapter layer, runs staged LLM checks and persists
both machine-readable and human-readable reports.
"""

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.config import get_settings
from app.domain.enums import LLMCheckStage, ReportStatus
from app.prompts.factory import PromptBuilderRegistry
from app.rag.retrieval_adapter import RetrievalAdapter
from app.repositories.audit_repository import AuditRepository
from app.repositories.visit_repository import VisitRepository
from app.schemas.llm import StageCheckResult
from app.schemas.retrieval import RetrievalQuery
from app.schemas.visit import VisitPreprocessResult
from app.services.diagnosis import DiagnosisContextExtractor
from app.services.heuristic_flags import VisitHeuristicFlagger
from app.services.llm_client import LLMClient, OpenAILLMClient
from app.services.renderers import VisitRenderer
from app.services.report_builder import AuditReportBuilder
from app.services.visit_classifier import VisitTypeClassifier
from app.services.visit_normalizer import VisitNormalizer
from app.utils.logging import get_logger


log = get_logger(__name__)


@dataclass(slots=True)
class VisitAuditResult:
    """Identifiers of created DB entities for one processed visit."""

    visit_id: str
    report_id: str
    status: str


class VisitAuditPipeline:
    """Coordinates all steps required to audit a single visit entry."""

    def __init__(
        self,
        session: Session,
        retrieval_adapter: RetrievalAdapter,
        llm_client: LLMClient | None = None,
        prompt_registry: PromptBuilderRegistry | None = None,
    ) -> None:
        """Create pipeline with pluggable retrieval and LLM dependencies."""
        self.session = session
        self.settings = get_settings()
        self.retrieval_adapter = retrieval_adapter
        self.llm_client = llm_client or OpenAILLMClient()
        self.prompt_registry = prompt_registry or PromptBuilderRegistry()

        self.visit_repo = VisitRepository(session)
        self.audit_repo = AuditRepository(session)

        self.normalizer = VisitNormalizer()
        self.flagger = VisitHeuristicFlagger()
        self.classifier = VisitTypeClassifier()
        self.dx_extractor = DiagnosisContextExtractor()
        self.renderer = VisitRenderer()
        self.report_builder = AuditReportBuilder()

    def process_one(self, raw_visit: dict[str, Any], external_id: str | None = None) -> VisitAuditResult:
        """Run end-to-end audit for one raw visit payload."""
        total_started = perf_counter()
        log.info("Visit audit started | external_id=%s", external_id)

        step_started = perf_counter()
        visit = self.visit_repo.create_raw_visit(raw_json=raw_visit, external_id=external_id)
        log.info(
            "Visit audit step done | step=create_raw_visit | visit_id=%s | elapsed_ms=%.1f",
            visit.id,
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        preprocess = self._preprocess(raw_visit)
        log.info(
            "Visit audit step done | step=preprocess | visit_id=%s | flags=%s | icd10=%s | visit_type=%s | elapsed_ms=%.1f",
            visit.id,
            len(preprocess.flags),
            len(preprocess.icd10_codes),
            preprocess.classification.visit_type.value,
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        readable_visit = self.renderer.to_markdown(preprocess.canonical_visit)
        self.visit_repo.update_preprocessed(
            visit=visit,
            normalized_json=preprocess.normalized_json,
            readable_render=readable_visit,
            visit_type=preprocess.classification.visit_type,
            specialty=preprocess.specialty,
            patient_age=preprocess.age,
            icd10_codes=preprocess.icd10_codes,
            extraction_flags=[flag.code for flag in preprocess.flags],
            metadata_json={
                "classification_confidence": preprocess.classification.confidence,
                "classification_reasons": preprocess.classification.reasons,
            },
        )
        log.info(
            "Visit audit step done | step=store_preprocessed_visit | visit_id=%s | elapsed_ms=%.1f",
            visit.id,
            (perf_counter() - step_started) * 1000,
        )

        stage_results: list[StageCheckResult] = []
        llm_trace: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []

        for stage in self.prompt_registry.stages():
            stage_started = perf_counter()
            log.info("Visit audit stage started | visit_id=%s | stage=%s", visit.id, stage.value)
            # Retrieval adapter hides storage/index internals from audit logic.
            retrieval_query = RetrievalQuery(
                diagnosis_codes=preprocess.icd10_codes,
                visit_type=preprocess.classification.visit_type,
                specialty=preprocess.specialty,
                section_targets=self._section_targets_for_stage(stage),
                requested_check_type=stage,
                max_chunks=self.settings.retrieval_top_k,
            )
            step_started = perf_counter()
            context = self.retrieval_adapter.retrieve_context(retrieval_query)
            log.info(
                "Visit audit stage step done | visit_id=%s | stage=%s | step=retrieve_context | refs=%s | elapsed_ms=%.1f",
                visit.id,
                stage.value,
                len(context.references_metadata),
                (perf_counter() - step_started) * 1000,
            )
            references.extend(context.references_metadata)

            # Stage-specific prompt builders keep prompting logic modular.
            step_started = perf_counter()
            builder = self.prompt_registry.get(stage)
            prompt = builder.build(
                visit=preprocess.canonical_visit,
                retrieval_context=context,
                prompt_conditions=preprocess.classification.prompt_conditions,
            )
            log.info(
                "Visit audit stage step done | visit_id=%s | stage=%s | step=build_prompt | elapsed_ms=%.1f",
                visit.id,
                stage.value,
                (perf_counter() - step_started) * 1000,
            )

            step_started = perf_counter()
            llm_resp = self.llm_client.generate(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                model=self.settings.llm_model,
            )
            log.info(
                "Visit audit stage step done | visit_id=%s | stage=%s | step=llm_call | latency_ms=%s | elapsed_ms=%.1f",
                visit.id,
                stage.value,
                llm_resp.latency_ms,
                (perf_counter() - step_started) * 1000,
            )

            step_started = perf_counter()
            stage_result = self._parse_stage_result(stage, llm_resp.text, llm_resp.raw)
            stage_result.token_usage = llm_resp.token_usage
            stage_result.latency_ms = llm_resp.latency_ms
            log.info(
                "Visit audit stage step done | visit_id=%s | stage=%s | step=parse_result | status=%s | findings=%s | elapsed_ms=%.1f",
                visit.id,
                stage.value,
                stage_result.status,
                len(stage_result.findings),
                (perf_counter() - step_started) * 1000,
            )

            stage_results.append(stage_result)
            llm_trace.append(
                {
                    "stage": stage.value,
                    "prompt_version": prompt.prompt_version,
                    "model": self.settings.llm_model,
                    "input_payload": {
                        "system_prompt": prompt.system_prompt,
                        "user_prompt": prompt.user_prompt,
                    },
                    "output_payload": llm_resp.raw,
                    "token_usage": llm_resp.token_usage,
                    "latency_ms": llm_resp.latency_ms,
                }
            )
            step_started = perf_counter()
            self.audit_repo.create_llm_history(
                visit=visit,
                stage=stage.value,
                prompt_version=prompt.prompt_version,
                model=self.settings.llm_model,
                input_payload={
                    "system_prompt": prompt.system_prompt,
                    "user_prompt": prompt.user_prompt,
                },
                output_payload=llm_resp.raw,
                latency_ms=llm_resp.latency_ms,
                token_usage_json=llm_resp.token_usage,
                status=stage_result.status,
            )
            log.info(
                "Visit audit stage step done | visit_id=%s | stage=%s | step=store_llm_history | elapsed_ms=%.1f",
                visit.id,
                stage.value,
                (perf_counter() - step_started) * 1000,
            )
            log.info(
                "Visit audit stage completed | visit_id=%s | stage=%s | total_stage_elapsed_ms=%.1f",
                visit.id,
                stage.value,
                (perf_counter() - stage_started) * 1000,
            )

        step_started = perf_counter()
        report_payload = self.report_builder.build(
            classification=preprocess.classification,
            heuristic_flags=preprocess.flags,
            stage_results=stage_results,
            references=references,
        )
        report_text = self.report_builder.to_text(report_payload)
        log.info(
            "Visit audit step done | step=build_report | visit_id=%s | human_review=%s | elapsed_ms=%.1f",
            visit.id,
            report_payload.human_review_required,
            (perf_counter() - step_started) * 1000,
        )

        status = ReportStatus.READY if not report_payload.human_review_required else ReportStatus.PARTIAL
        step_started = perf_counter()
        report_row = self.audit_repo.create_report(
            visit=visit,
            report_json=report_payload.model_dump(mode="json"),
            report_text=report_text,
            status=status.value,
            scores_json=report_payload.scores,
            llm_trace_metadata={"stages": llm_trace},
            readable_visit_card=readable_visit,
        )
        log.info(
            "Visit audit step done | step=store_report | visit_id=%s | report_id=%s | elapsed_ms=%.1f",
            visit.id,
            report_row.id,
            (perf_counter() - step_started) * 1000,
        )

        step_started = perf_counter()
        self.session.flush()
        log.info(
            "Visit audit step done | step=flush | visit_id=%s | report_id=%s | elapsed_ms=%.1f",
            visit.id,
            report_row.id,
            (perf_counter() - step_started) * 1000,
        )
        log.info(
            "Visit processed | visit_id=%s | report_id=%s | status=%s | total_elapsed_ms=%.1f",
            visit.id,
            report_row.id,
            status.value,
            (perf_counter() - total_started) * 1000,
        )
        return VisitAuditResult(visit_id=str(visit.id), report_id=str(report_row.id), status=status.value)

    def _preprocess(self, raw_visit: dict[str, Any]) -> VisitPreprocessResult:
        """Canonicalize visit and collect deterministic features before LLM calls."""
        canonical = self.normalizer.normalize(raw_visit)
        flags = self.flagger.generate_flags(canonical)
        classification = self.classifier.classify(canonical)
        icd10_codes = self.dx_extractor.extract_icd10_codes(canonical)

        specialty = str(canonical.admin.get("specialty", "") or canonical.meta.get("specialty", "")).strip() or None
        age = self._extract_age(canonical.patient)

        normalized_json = canonical.model_dump(mode="json")

        return VisitPreprocessResult(
            canonical_visit=canonical,
            flags=flags,
            classification=classification,
            icd10_codes=icd10_codes,
            specialty=specialty,
            age=age,
            normalized_json=normalized_json,
        )

    def _extract_age(self, patient: dict[str, Any]) -> int | None:
        """Extract patient age from common canonical keys."""
        for key in ("age", "возраст", "patient_age"):
            if key in patient:
                try:
                    return int(patient[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _section_targets_for_stage(self, stage: LLMCheckStage) -> list[str]:
        """Map check stage to high-signal guideline section types for retrieval."""
        mapping = {
            LLMCheckStage.FORMAL_STRUCTURE_CHECK: ["brief_info", "diagnostics", "treatment"],
            LLMCheckStage.DIAGNOSIS_CONSISTENCY_CHECK: ["diagnostics", "brief_info"],
            LLMCheckStage.MANAGEMENT_CONSISTENCY_CHECK: ["treatment", "diagnostics"],
            LLMCheckStage.FOLLOWUP_CHECK: ["prevention_followup", "additional_info"],
            LLMCheckStage.DOCUMENTATION_QUALITY_CHECK: ["quality_criteria", "organization_of_care"],
        }
        return mapping.get(stage, ["unknown"])

    def _parse_stage_result(self, stage: LLMCheckStage, text: str, raw: dict[str, Any]) -> StageCheckResult:
        """Convert LLM JSON payload into typed stage result with safe fallback."""
        try:
            parsed = json.loads(text)
            findings = parsed.get("findings") if isinstance(parsed, dict) else []
            if not isinstance(findings, list):
                findings = []
            return StageCheckResult(
                stage=stage,
                status=str(parsed.get("status", "ok")),
                summary=str(parsed.get("summary", "")),
                findings=findings,
                raw_response=raw,
            )
        except Exception:
            return StageCheckResult(
                stage=stage,
                status="parse_error",
                summary="LLM output could not be parsed as JSON",
                findings=[{"code": "llm_parse_error", "message": text[:500], "severity": "warning"}],
                raw_response=raw,
            )
