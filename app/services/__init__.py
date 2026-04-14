"""Service-layer exports used by pipelines and integration entry points."""

from app.services.diagnosis import DiagnosisContextExtractor
from app.services.embedding import (
    EmbeddingsAdapter,
    EmbeddingProvider,
    OllamaEmbeddingsBackend,
    OllamaEmbeddings,
    OpenAIEmbeddingsBackend,
    OpenAIEmbeddingProvider,
    RawEmbeddingBackend,
    create_embedding_backend,
    create_embedding_provider,
)
from app.services.heuristic_flags import VisitHeuristicFlagger
from app.services.llm_client import (
    LLMAdapter,
    LLMClient,
    OllamaChatBackend,
    OllamaLLMClient,
    OpenAIChatBackend,
    OpenAILLMClient,
    RawLLMBackend,
    create_llm_backend,
    create_llm_client,
)
from app.services.normative_prompt_rules import NormativePromptRuleProvider
from app.services.renderers import VisitRenderer
from app.services.report_builder import AuditReportBuilder
from app.services.visit_classifier import VisitTypeClassifier
from app.services.visit_normalizer import VisitNormalizer

__all__ = [
    "EmbeddingProvider",
    "RawEmbeddingBackend",
    "EmbeddingsAdapter",
    "OpenAIEmbeddingsBackend",
    "OllamaEmbeddingsBackend",
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddings",
    "create_embedding_backend",
    "create_embedding_provider",
    "LLMClient",
    "RawLLMBackend",
    "LLMAdapter",
    "OpenAIChatBackend",
    "OllamaChatBackend",
    "OpenAILLMClient",
    "OllamaLLMClient",
    "create_llm_backend",
    "create_llm_client",
    "NormativePromptRuleProvider",
    "VisitNormalizer",
    "VisitHeuristicFlagger",
    "VisitTypeClassifier",
    "DiagnosisContextExtractor",
    "VisitRenderer",
    "AuditReportBuilder",
]
