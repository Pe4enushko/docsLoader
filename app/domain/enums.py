from __future__ import annotations

from enum import Enum


class DocumentKind(str, Enum):
    GUIDELINE = "guideline"
    NORMATIVE = "normative"


class DocumentStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class SectionType(str, Enum):
    TITLE_PAGE = "title_page"
    TOC = "toc"
    ABBREVIATIONS = "abbreviations"
    DEFINITIONS = "definitions"
    BRIEF_INFO = "brief_info"
    DIAGNOSTICS = "diagnostics"
    TREATMENT = "treatment"
    REHABILITATION = "rehabilitation"
    PREVENTION_FOLLOWUP = "prevention_followup"
    ORGANIZATION_OF_CARE = "organization_of_care"
    ADDITIONAL_INFO = "additional_info"
    QUALITY_CRITERIA = "quality_criteria"
    BIBLIOGRAPHY = "bibliography"
    APPENDIX = "appendix"
    UNKNOWN = "unknown"


class SectionItemType(str, Enum):
    RECOMMENDATION = "recommendation"
    TABLE = "table"
    ALGORITHM = "algorithm"
    APPENDIX = "appendix"
    RAW_TEXT = "raw_text"


class ChunkType(str, Enum):
    NARRATIVE = "narrative"
    RECOMMENDATION = "recommendation"
    TABLE = "table"
    ALGORITHM = "algorithm"
    APPENDIX = "appendix"


class RuleType(str, Enum):
    DIAGNOSTIC = "diagnostic"
    MANAGEMENT = "management"
    FOLLOWUP = "followup"
    DOCUMENTATION = "documentation"
    QUALITY = "quality"
    OTHER = "other"


class VisitType(str, Enum):
    PRIMARY = "primary"
    REPEAT = "repeat"
    PROPHYLACTIC = "prophylactic"
    UNKNOWN = "unknown"


class ReportStatus(str, Enum):
    READY = "ready"
    FAILED = "failed"
    PARTIAL = "partial"


class LLMCheckStage(str, Enum):
    FORMAL_STRUCTURE_CHECK = "formal_structure_check"
    DIAGNOSIS_CONSISTENCY_CHECK = "diagnosis_consistency_check"
    MANAGEMENT_CONSISTENCY_CHECK = "management_consistency_check"
    FOLLOWUP_CHECK = "followup_check"
    DOCUMENTATION_QUALITY_CHECK = "documentation_quality_check"


class JobType(str, Enum):
    INGEST_DOCUMENT = "ingest_document"
    AUDIT_VISIT = "audit_visit"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
