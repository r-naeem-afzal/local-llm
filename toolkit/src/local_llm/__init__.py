"""local_llm — delegate mechanical work to a local model instead of a paid API.

Point it at any OpenAI-compatible server (Bionic/LM Studio, llama.cpp, vLLM, Ollama's
compat endpoint) with two environment variables:

    LOCAL_LLM_URL=http://localhost:1234/v1
    LOCAL_LLM_MODEL=qwen/qwen3-14b

Then build the toolkit and ask it for the service you want:

    import asyncio
    from local_llm import Toolkit

    toolkit = Toolkit()

    # Typed structured output — the JSON schema is generated from the pydantic model
    # and the reply is validated against it, so a malformed claim raises rather than
    # returning a subtly wrong dict.
    extraction = asyncio.run(
        toolkit.claim_extractor.extract(
            "https://example.com/article",
            "What does this say about pricing?",
        )
    )
    for claim in extraction.claims:
        print(claim.importance, claim.claim)

`Toolkit` is the composition root: it wires the concrete classes together and is the only
place that knows history is stored in SQLite and live progress in files. Everything else
takes its collaborators through its constructor, so any piece can be used on its own or
driven by a fake.

Every call is recorded with its prompt, response, reasoning, token usage and timing, and
streams live progress while it runs, so the dashboard can show exactly what the model was
asked and what it said.
"""

from .client import (
    CompletionClient,
    JsonResponseParser,
    LocalLLMClient,
    LocalLLMError,
    ProgressReporter,
    StreamAccumulator,
    ThinkingStripper,
)
from .config import DatabaseConfig, Settings
from .database import (
    BACKENDS,
    DatabaseBackend,
    MySqlBackend,
    SqliteBackend,
    build_backend,
)
from .container import Toolkit
from .routing import ModelClassifier, ModelProfile, ModelRole, ModelRouter
from .extract import (
    ArticleExtractor,
    Claim,
    ClaimExtractor,
    Extraction,
    ExtractorChain,
    Page,
    PageFetcher,
    PromptLibrary,
    IndexRanking,
    RankedIndex,
    RankedResult,
    Ranking,
    RegexExtractor,
    ResultRanker,
    SourceExtraction,
    TrafilaturaExtractor,
)
from .agents import (
    ActiveAgent,
    AgentActivity,
    AgentActivityReader,
    AgentInvocation,
    AgentTranscriptScanner,
)
from .monitor import (
    AgentUsage,
    ClaudeUsage,
    ClaudeUsageReader,
    GpuInfo,
    GpuProbe,
    HostInfo,
    HostProbe,
    LmsCommandRunner,
    LoadedModel,
    ModelRegistry,
    ModelServerProbe,
    SystemMonitor,
    TranscriptLocator,
    TranscriptParser,
    UsageRecord,
)
from .store import (
    CallPayload,
    CallRecord,
    CallRepository,
    ConnectionProvider,
    FileLiveProgressStore,
    LiveProgressStore,
    RetentionService,
    SchemaMigrator,
    SqlCallRepository,
)

__version__ = "0.1.0"

__all__ = [
    # composition root — the usual entry point
    "Toolkit",
    "Settings",
    "ModelRouter",
    "ModelRole",
    "ModelProfile",
    "ModelClassifier",
    "DatabaseConfig",
    # model access
    "CompletionClient",
    "LocalLLMClient",
    "LocalLLMError",
    "StreamAccumulator",
    "ProgressReporter",
    "ThinkingStripper",
    "JsonResponseParser",
    # extraction
    "ClaimExtractor",
    "ResultRanker",
    "PageFetcher",
    "PromptLibrary",
    "ArticleExtractor",
    "TrafilaturaExtractor",
    "RegexExtractor",
    "ExtractorChain",
    "Claim",
    "Extraction",
    "SourceExtraction",
    "Page",
    "Ranking",
    "RankedResult",
    "RankedIndex",
    "IndexRanking",
    # persistence
    "CallRepository",
    "SqlCallRepository",
    "LiveProgressStore",
    "FileLiveProgressStore",
    "ConnectionProvider",
    "SchemaMigrator",
    # database engines
    "DatabaseBackend",
    "SqliteBackend",
    "MySqlBackend",
    "build_backend",
    "BACKENDS",
    "RetentionService",
    "CallRecord",
    "CallPayload",
    # monitoring
    "SystemMonitor",
    "GpuProbe",
    "HostProbe",
    "ModelRegistry",
    "ModelServerProbe",
    "LmsCommandRunner",
    "ClaudeUsageReader",
    "TranscriptParser",
    "TranscriptLocator",
    "GpuInfo",
    "HostInfo",
    "LoadedModel",
    "ClaudeUsage",
    "AgentUsage",
    "UsageRecord",
    # live Claude agent activity
    "AgentActivityReader",
    "AgentTranscriptScanner",
    "AgentActivity",
    "ActiveAgent",
    "AgentInvocation",
    "__version__",
]
