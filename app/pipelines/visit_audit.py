from __future__ import annotations

"""LangGraph-powered pipeline for auditing one medical visit record.

Implementation notes:
- each business step is represented as an explicit graph node,
- intermediate artifacts are stored in graph state (`VisitAuditState`),
- graph execution stays deterministic and observable for debugging.
"""

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.config import get_settings
from app.domain.enums import LLMCheckStage, ReportStatus
from app.models.visit import AuditReport, VisitRecord
from app.prompts.factory import PromptBuilderRegistry
from app.rag.retrieval_adapter import RetrievalAdapter
from app.repositories.audit_repository import AuditRepository
from app.repositories.visit_repository import VisitRepository
from app.schemas.llm import StageCheckResult
from app.schemas.report import AuditReportPayload
from app.schemas.retrieval import RetrievalQuery
from app.schemas.visit import VisitPreprocessResult
from app.services.diagnosis import DiagnosisContextExtractor
from app.services.heuristic_flags import VisitHeuristicFlagger
from app.services.llm_client import LLMClient, create_llm_client
from app.services.normative_prompt_rules import NormativePromptRuleProvider
from app.services.renderers import VisitRenderer
from app.services.report_builder import AuditReportBuilder
from app.services.visit_classifier import VisitTypeClassifier
from app.services.visit_normalizer import VisitNormalizer
from app.utils.logging import get_pipeline_logger


log = get_pipeline_logger(__name__, "visit_audit_pipeline.log")


@dataclass(slots=True)
class VisitAuditResult:
    """Identifiers of created DB entities for one processed visit."""

    visit_id: UUID
    report_id: UUID
    status: str


class VisitAuditState(TypedDict, total=False):
    """State container passed between LangGraph nodes.

    What is stored:
    - input artifacts: original payload and external id,
    - deterministic preprocessing outputs: canonical visit, flags, classification, ICD list,
    - stage artifacts: per-stage findings, retrieval references, LLM traces,
    - final artifacts: structured report, human-readable report, DB row ids,
    - telemetry: compact node-level summaries for debugging.
    """

    # Original visit JSON passed to pipeline.
    raw_visit: dict[str, Any]
    # Stable external id from source system (if available).
    external_id: str | None
    # Pipeline start timestamp for total execution metrics.
    started_at: float

    # DB row created from raw visit payload.
    visit: VisitRecord
    # Deterministic preprocessing bundle shared by all stages.
    preprocess: VisitPreprocessResult
    # Human-oriented markdown rendering of canonical visit.
    readable_visit: str

    # Accumulated parsed results from all LLM stages.
    stage_results: list[StageCheckResult]
    # Raw per-stage model traces (prompts, outputs, usage, latency).
    llm_trace: list[dict[str, Any]]
    # References returned by retrieval adapter for evidence traceability.
    references: list[dict[str, Any]]

    # Final typed report payload used for persistence and API responses.
    report_payload: AuditReportPayload
    # Human-readable report text generated from structured payload.
    report_text: str
    # Final report status (`ready` or `partial` depending on review requirement).
    report_status: ReportStatus
    # Persisted DB row of final report.
    report_row: AuditReport

    # Compact result returned to caller.
    result: VisitAuditResult
    # Per-node telemetry payloads (elapsed time, counts, statuses).
    node_results: dict[str, Any]


class VisitAuditPipeline:
    """Coordinates full visit-audit flow through a LangGraph state machine."""

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
        self.llm_client = llm_client or create_llm_client()
        self.prompt_registry = prompt_registry or PromptBuilderRegistry()
        log.info(
            "Visit audit LLM provider initialized | provider=%s",
            self.llm_client.__class__.__name__,
        )

        self.visit_repo = VisitRepository(session)
        self.audit_repo = AuditRepository(session)

        self.normalizer = VisitNormalizer()
        self.flagger = VisitHeuristicFlagger()
        self.classifier = VisitTypeClassifier()
        self.dx_extractor = DiagnosisContextExtractor()
        self.normative_rules = NormativePromptRuleProvider()
        self.renderer = VisitRenderer()
        self.report_builder = AuditReportBuilder()

        # Graph is compiled once and reused for each visit.
        self._graph = self._build_graph()

    def get_graph(self):
        """Return compiled LangGraph app instance for debugging/visualization."""
        return self._graph

    def process_one(self, raw_visit: dict[str, Any], external_id: str | None = None) -> VisitAuditResult:
        """Run graph pipeline and return only final compact result."""
        result, _state = self.process_one_with_state(raw_visit=raw_visit, external_id=external_id)
        return result

    def process_one_with_state(
        self,
        raw_visit: dict[str, Any],
        external_id: str | None = None,
    ) -> tuple[VisitAuditResult, VisitAuditState]:
        """Run graph pipeline and return final result with full state snapshot."""
        initial_state: VisitAuditState = {
            "raw_visit": raw_visit,
            "external_id": external_id,
            "started_at": perf_counter(),
            "stage_results": [],
            "llm_trace": [],
            "references": [],
            "node_results": {},
        }
        log.info("Visit audit started | external_id=%s", external_id)
        final_state = self._graph.invoke(initial_state)
        result = self._require_state(final_state, "result", "finalize")
        return result, final_state

    def _build_graph(self):
        """Build ordered LangGraph with explicit domain nodes."""
        graph = StateGraph(VisitAuditState)

        # Core visit lifecycle nodes.
        graph.add_node("create_raw_visit", self._node_create_raw_visit)
        graph.add_node("preprocess_visit", self._node_preprocess_visit)
        graph.add_node("store_preprocessed_visit", self._node_store_preprocessed_visit)

        # Stage-specific audit nodes.
        graph.add_node("stage_formal_structure", self._node_stage_formal_structure)
        graph.add_node("stage_diagnosis_consistency", self._node_stage_diagnosis_consistency)
        graph.add_node("stage_management_consistency", self._node_stage_management_consistency)
        graph.add_node("stage_followup", self._node_stage_followup)
        graph.add_node("stage_documentation_quality", self._node_stage_documentation_quality)

        # Finalization nodes.
        graph.add_node("build_report", self._node_build_report)
        graph.add_node("store_report", self._node_store_report)
        graph.add_node("finalize", self._node_finalize)

        # Sequential edges explicitly document pipeline order.
        graph.add_edge(START, "create_raw_visit")
        graph.add_edge("create_raw_visit", "preprocess_visit")
        graph.add_edge("preprocess_visit", "store_preprocessed_visit")
        graph.add_edge("store_preprocessed_visit", "stage_formal_structure")
        graph.add_edge("stage_formal_structure", "stage_diagnosis_consistency")
        graph.add_edge("stage_diagnosis_consistency", "stage_management_consistency")
        graph.add_edge("stage_management_consistency", "stage_followup")
        graph.add_edge("stage_followup", "stage_documentation_quality")
        graph.add_edge("stage_documentation_quality", "build_report")
        graph.add_edge("build_report", "store_report")
        graph.add_edge("store_report", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _node_create_raw_visit(self, state: VisitAuditState) -> VisitAuditState:
        """Node 1: persist raw visit payload as starting record.

        Checks/logic:
        - no clinical validation yet; only durability and id binding.
        Adds:
        - `visit` DB entity to state,
        - node telemetry with created `visit_id`.
        """
        step_started = perf_counter()
        visit = self.visit_repo.create_raw_visit(
            raw_json=self._require_state(state, "raw_visit", "create_raw_visit"),
            external_id=state.get("external_id"),
        )
        elapsed_ms = (perf_counter() - step_started) * 1000
        log.info(
            "Visit audit node done | node=create_raw_visit | visit_id=%s | elapsed_ms=%.1f",
            visit.id,
            elapsed_ms,
        )
        return {
            "visit": visit,
            "node_results": self._record_node_result(
                state,
                "create_raw_visit",
                {
                    "visit_id": str(visit.id),
                    "external_id": state.get("external_id"),
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            ),
        }

    def _node_preprocess_visit(self, state: VisitAuditState) -> VisitAuditState:
        """Node 2: normalize visit, classify type and generate heuristic flags.

        Checks/logic:
        - canonical mapping of source payload to domain sections,
        - deterministic heuristic checks (missing fields, placeholders, mixups, etc.),
        - visit type classification (primary/repeat/prophylactic),
        - ICD-10 extraction for retrieval query construction.
        Adds:
        - `preprocess` bundle to state with canonical visit, flags and classification.
        """
        step_started = perf_counter()
        preprocess = self._preprocess(self._require_state(state, "raw_visit", "preprocess_visit"))
        elapsed_ms = (perf_counter() - step_started) * 1000
        log.info(
            "Visit audit node done | node=preprocess_visit | visit_type=%s | flags=%s | icd10=%s | elapsed_ms=%.1f",
            preprocess.classification.visit_type.value,
            len(preprocess.flags),
            len(preprocess.icd10_codes),
            elapsed_ms,
        )
        return {
            "preprocess": preprocess,
            "node_results": self._record_node_result(
                state,
                "preprocess_visit",
                {
                    "visit_type": preprocess.classification.visit_type.value,
                    "flags_count": len(preprocess.flags),
                    "icd10_count": len(preprocess.icd10_codes),
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            ),
        }

    def _node_store_preprocessed_visit(self, state: VisitAuditState) -> VisitAuditState:
        """Node 3: persist normalized card and physician-readable representation.

        Checks/logic:
        - converts canonical visit to markdown for manual physician review.
        Adds:
        - updates `visit_records` with normalized JSON, flags, visit_type, ICD list,
        - stores `readable_visit` snapshot in state.
        """
        step_started = perf_counter()
        visit = self._require_state(state, "visit", "store_preprocessed_visit")
        preprocess = self._require_state(state, "preprocess", "store_preprocessed_visit")
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
        elapsed_ms = (perf_counter() - step_started) * 1000
        log.info(
            "Visit audit node done | node=store_preprocessed_visit | visit_id=%s | elapsed_ms=%.1f",
            visit.id,
            elapsed_ms,
        )
        return {
            "readable_visit": readable_visit,
            "node_results": self._record_node_result(
                state,
                "store_preprocessed_visit",
                {
                    "visit_id": str(visit.id),
                    "readable_chars": len(readable_visit),
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            ),
        }

    def _node_stage_formal_structure(self, state: VisitAuditState) -> VisitAuditState:
        """Node 4: formal structure check stage.

        What is checked:
        - structural completeness and consistency of note sections,
        - obvious contradictions between note structure and expected visit type.
        What is added:
        - stage findings + references + llm trace into state accumulators.
        """
        return self._run_stage_node(
            state,
            stage=LLMCheckStage.FORMAL_STRUCTURE_CHECK,
            node_name="stage_formal_structure",
        )

    def _node_stage_diagnosis_consistency(self, state: VisitAuditState) -> VisitAuditState:
        """Node 5: diagnosis consistency check stage.

        What is checked:
        - diagnosis/ICD coherence with subjective and objective findings,
        - adequacy of diagnosis specificity relative to available evidence.
        What is added:
        - parsed findings and provenance for diagnosis quality block.
        """
        return self._run_stage_node(
            state,
            stage=LLMCheckStage.DIAGNOSIS_CONSISTENCY_CHECK,
            node_name="stage_diagnosis_consistency",
        )

    def _node_stage_management_consistency(self, state: VisitAuditState) -> VisitAuditState:
        """Node 6: management consistency check stage.

        What is checked:
        - treatment/diagnostic plan consistency with diagnosis and recommendations,
        - evidence-based adequacy of selected management tactics.
        What is added:
        - stage-level management findings and supporting references.
        """
        return self._run_stage_node(
            state,
            stage=LLMCheckStage.MANAGEMENT_CONSISTENCY_CHECK,
            node_name="stage_management_consistency",
        )

    def _node_stage_followup(self, state: VisitAuditState) -> VisitAuditState:
        """Node 7: follow-up adequacy check stage.

        What is checked:
        - follow-up timing and dynamic monitoring adequacy,
        - presence/quality of next-step and re-evaluation instructions.
        What is added:
        - follow-up findings with trace metadata.
        """
        return self._run_stage_node(
            state,
            stage=LLMCheckStage.FOLLOWUP_CHECK,
            node_name="stage_followup",
        )

    def _node_stage_documentation_quality(self, state: VisitAuditState) -> VisitAuditState:
        """Node 8: documentation quality check stage.

        What is checked:
        - clarity and auditability of documentation,
        - medico-legal/document quality gaps and ambiguity markers.
        What is added:
        - documentation quality findings for final report merge.
        """
        return self._run_stage_node(
            state,
            stage=LLMCheckStage.DOCUMENTATION_QUALITY_CHECK,
            node_name="stage_documentation_quality",
        )

    def _node_build_report(self, state: VisitAuditState) -> VisitAuditState:
        """Node 9: merge all findings into final report artifacts.

        Checks/logic:
        - merges deterministic and LLM findings into unified section-wise report,
        - computes score aggregates and human-review requirement.
        Adds:
        - `report_payload`, `report_text`, `report_status` into state.
        """
        step_started = perf_counter()
        preprocess = self._require_state(state, "preprocess", "build_report")
        stage_results = list(state.get("stage_results", []))
        references = list(state.get("references", []))

        report_payload = self.report_builder.build(
            classification=preprocess.classification,
            heuristic_flags=preprocess.flags,
            stage_results=stage_results,
            references=references,
        )
        report_text = self.report_builder.to_text(report_payload)
        report_status = ReportStatus.READY if not report_payload.human_review_required else ReportStatus.PARTIAL
        elapsed_ms = (perf_counter() - step_started) * 1000

        log.info(
            "Visit audit node done | node=build_report | human_review=%s | elapsed_ms=%.1f",
            report_payload.human_review_required,
            elapsed_ms,
        )
        return {
            "report_payload": report_payload,
            "report_text": report_text,
            "report_status": report_status,
            "node_results": self._record_node_result(
                state,
                "build_report",
                {
                    "human_review_required": report_payload.human_review_required,
                    "scores": report_payload.scores,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            ),
        }

    def _node_store_report(self, state: VisitAuditState) -> VisitAuditState:
        """Node 10: persist report, flush session and prepare final return object.

        Checks/logic:
        - writes final report JSON/text and trace metadata to DB.
        Adds:
        - `report_row` and `result` (visit/report ids) into state.
        """
        step_started = perf_counter()
        visit = self._require_state(state, "visit", "store_report")
        report_payload = self._require_state(state, "report_payload", "store_report")
        report_text = self._require_state(state, "report_text", "store_report")
        report_status = self._require_state(state, "report_status", "store_report")
        readable_visit = self._require_state(state, "readable_visit", "store_report")
        llm_trace = list(state.get("llm_trace", []))

        report_row = self.audit_repo.create_report(
            visit=visit,
            report_json=report_payload.model_dump(mode="json"),
            report_text=report_text,
            status=report_status.value,
            scores_json=report_payload.scores,
            llm_trace_metadata={"stages": llm_trace},
            readable_visit_card=readable_visit,
        )
        self.session.flush()

        result = VisitAuditResult(visit_id=visit.id, report_id=report_row.id, status=report_status.value)
        elapsed_ms = (perf_counter() - step_started) * 1000
        log.info(
            "Visit audit node done | node=store_report | visit_id=%s | report_id=%s | elapsed_ms=%.1f",
            visit.id,
            report_row.id,
            elapsed_ms,
        )
        return {
            "report_row": report_row,
            "result": result,
            "node_results": self._record_node_result(
                state,
                "store_report",
                {
                    "visit_id": str(visit.id),
                    "report_id": str(report_row.id),
                    "status": report_status.value,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            ),
        }

    def _node_finalize(self, state: VisitAuditState) -> VisitAuditState:
        """Node 11: finalize telemetry summary for the entire graph run.

        Checks/logic:
        - no clinical checks; only run-level telemetry completion.
        Adds:
        - total elapsed time in `node_results.finalize`.
        """
        result = self._require_state(state, "result", "finalize")
        started_at = state.get("started_at", perf_counter())
        total_elapsed_ms = (perf_counter() - started_at) * 1000
        log.info(
            "Visit processed | visit_id=%s | report_id=%s | status=%s | total_elapsed_ms=%.1f",
            result.visit_id,
            result.report_id,
            result.status,
            total_elapsed_ms,
        )
        return {
            "node_results": self._record_node_result(
                state,
                "finalize",
                {"total_elapsed_ms": round(total_elapsed_ms, 1)},
            )
        }

    def _run_stage_node(self, state: VisitAuditState, *, stage: LLMCheckStage, node_name: str) -> VisitAuditState:
        """Execute one LLM audit stage as an isolated graph node.

        This node performs retrieval, prompt build, LLM call, parsing and
        persistence of stage-level LLM history. All artifacts are appended to
        graph state arrays.

        Added state artifacts per stage:
        - `stage_results += [StageCheckResult]`
        - `references += retrieval_context.references_metadata`
        - `llm_trace += [{prompt/model/output/usage/latency}]`
        """
        step_started = perf_counter()
        visit = self._require_state(state, "visit", node_name)
        preprocess = self._require_state(state, "preprocess", node_name)

        # 1) Retrieve context relevant for this exact stage.
        retrieval_query = RetrievalQuery(
            diagnosis_codes=preprocess.icd10_codes,
            visit_type=preprocess.classification.visit_type,
            specialty=preprocess.specialty,
            section_targets=self._section_targets_for_stage(stage),
            requested_check_type=stage,
            max_chunks=self.settings.retrieval_top_k,
        )
        context = self.retrieval_adapter.retrieve_context(retrieval_query)

        # 2) Build stage prompt with visit snapshot, conditions and retrieved context.
        prompt_conditions = self._build_stage_conditions(stage=stage, preprocess=preprocess)
        builder = self.prompt_registry.get(stage)
        prompt = builder.build(
            visit=preprocess.canonical_visit,
            retrieval_context=context,
            prompt_conditions=prompt_conditions,
        )

        # 3) Call LLM in strict JSON mode so parser receives structured output.
        llm_resp = self.llm_client.generate(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            model=self.settings.llm_model,
            json_mode=True,
        )

        # 4) Parse payload and normalize stage outcome.
        stage_result = self._parse_stage_result(stage, llm_resp.text, llm_resp.raw)
        stage_result.token_usage = llm_resp.token_usage
        stage_result.latency_ms = llm_resp.latency_ms

        # 5) Persist stage trace for auditability.
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

        stage_results = [*state.get("stage_results", []), stage_result]
        references = [*state.get("references", []), *context.references_metadata]
        injected_reference_count = 0
        if stage == LLMCheckStage.FORMAL_STRUCTURE_CHECK:
            formal_rules = self.normative_rules.get_applicable_rules(
                visit_type=preprocess.classification.visit_type,
                specialty=preprocess.specialty,
                patient_age=preprocess.age,
            )
            injected_refs = self.normative_rules.to_reference_metadata(formal_rules)
            references.extend(injected_refs)
            injected_reference_count = len(injected_refs)
        llm_trace = [
            *state.get("llm_trace", []),
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
            },
        ]
        elapsed_ms = (perf_counter() - step_started) * 1000

        log.info(
            "Visit audit node done | node=%s | visit_id=%s | stage=%s | status=%s | findings=%s | refs=%s | elapsed_ms=%.1f",
            node_name,
            visit.id,
            stage.value,
            stage_result.status,
            len(stage_result.findings),
            len(context.references_metadata) + injected_reference_count,
            elapsed_ms,
        )
        return {
            "stage_results": stage_results,
            "references": references,
            "llm_trace": llm_trace,
            "node_results": self._record_node_result(
                state,
                node_name,
                {
                    "stage": stage.value,
                    "status": stage_result.status,
                    "findings_count": len(stage_result.findings),
                    "references_count": len(context.references_metadata) + injected_reference_count,
                    "latency_ms": llm_resp.latency_ms,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            ),
        }

    def _build_stage_conditions(self, *, stage: LLMCheckStage, preprocess: VisitPreprocessResult) -> list[str]:
        """Compose prompt conditions for stage with optional normative injection."""
        conditions = list(preprocess.classification.prompt_conditions)
        if stage != LLMCheckStage.FORMAL_STRUCTURE_CHECK:
            return conditions

        applicable_rules = self.normative_rules.get_applicable_rules(
            visit_type=preprocess.classification.visit_type,
            specialty=preprocess.specialty,
            patient_age=preprocess.age,
        )
        if not applicable_rules:
            return conditions

        conditions.append("Нормативные правила формальной структуры (prompt-injection):")
        conditions.extend(self.normative_rules.render_for_prompt(applicable_rules))
        return conditions

    def _preprocess(self, raw_visit: dict[str, Any]) -> VisitPreprocessResult:
        """Canonicalize visit and collect deterministic features before LLM calls.

        Output includes everything required to build retrieval queries and prompt
        conditions without invoking LLM:
        - canonical sections,
        - heuristic flags,
        - visit type + conditions,
        - ICD-10 codes, specialty, age.
        """
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

    def _record_node_result(self, state: VisitAuditState, node_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append/replace compact node artifact summary in state."""
        node_results = dict(state.get("node_results", {}))
        node_results[node_name] = payload
        return node_results

    def _require_state(self, state: VisitAuditState, key: str, node_name: str):
        """Fail fast when mandatory state artifact is missing."""
        if key not in state:
            raise RuntimeError(f"State key '{key}' is missing before node '{node_name}'")
        return state[key]
