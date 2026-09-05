"""The composition root — the one place that decides which concrete classes get used.

Every other class in this package takes its collaborators through its constructor and
never reaches for a global. That is what makes them testable, but something has to
actually *build* the object graph, and doing that inline at each call site would spread
the wiring everywhere.

`Toolkit` is that place. It is the only module that knows, for instance, that the
repository is backed by SQLite and that live progress lives in files. Swapping either is
a change here and nowhere else.

    from local_llm import Toolkit

    toolkit = Toolkit()
    extraction = await toolkit.claim_extractor.extract(url, question)

Dependencies are created lazily and cached, so importing the package does not touch the
disk or the GPU. That matters because a CLI asking one question — is the server up? —
should not pay to open a database and initialise NVML first.
"""

from __future__ import annotations

from functools import cached_property

from .agents import AgentActivityReader, AgentTranscriptScanner
from .client import LocalLLMClient
from .config import Settings
from .database import DatabaseBackend, build_backend
from .extract import ClaimExtractor, ExtractorChain, PageFetcher, PromptLibrary, ResultRanker
from .routing import ModelRouter
from .monitor import (
    ClaudeUsageReader,
    GpuProbe,
    HostProbe,
    LmsCommandRunner,
    ModelRegistry,
    ModelServerProbe,
    SystemMonitor,
    TranscriptLocator,
)
from .search import SearchService, UrlCanonicaliser
from .store import (
    ConnectionProvider,
    FileLiveProgressStore,
    RetentionService,
    SchemaMigrator,
    SqlCallRepository,
)


class Toolkit:
    """Builds and holds the wired-up object graph.

    `cached_property` is what makes each dependency lazy *and* a singleton within one
    Toolkit: the first access computes the value and replaces the property with it, so
    every later access returns the same instance. That single-instance behaviour is
    required, not incidental — two `ConnectionProvider`s would mean two sets of
    thread-local SQLite connections to the same file, doubling write contention for no
    reason.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        # Settings are built here if not supplied, which is the seam a test uses to point
        # the whole graph at a temporary directory and a fake endpoint.
        self._settings = settings or Settings()
        # Done once, here, rather than by every class that writes a file — so the "make
        # sure the folder exists" concern lives in exactly one place.
        self._settings.ensure_dirs()

    @property
    def settings(self) -> Settings:
        return self._settings

    # ── storage ──

    @cached_property
    def backend(self) -> DatabaseBackend:
        """The database engine, chosen by `settings.database.engine`.

        This is the only line in the package that decides which engine is in play. Every
        other class talks to the `DatabaseBackend` interface, so pointing the toolkit at
        MySQL instead of SQLite is a configuration change with no code change anywhere.
        """
        return build_backend(self._settings.database, self._settings.data_dir)

    @cached_property
    def migrator(self) -> SchemaMigrator:
        return SchemaMigrator(self.backend)

    @cached_property
    def connections(self) -> ConnectionProvider:
        return ConnectionProvider(self.backend, self.migrator)

    @cached_property
    def repository(self) -> SqlCallRepository:
        return SqlCallRepository(self.connections)

    @cached_property
    def live_store(self) -> FileLiveProgressStore:
        return FileLiveProgressStore(self._settings.live_dir)

    @cached_property
    def retention(self) -> RetentionService:
        return RetentionService(
            self.connections,
            self.repository,
            self._settings.payload_days,
            self._settings.max_payloads,
        )

    # ── model access ──

    @cached_property
    def client(self) -> LocalLLMClient:
        return LocalLLMClient(self._settings, self.repository, self.live_store)

    @cached_property
    def prompts(self) -> PromptLibrary:
        return PromptLibrary()

    @cached_property
    def page_fetcher(self) -> PageFetcher:
        return PageFetcher(self._settings, ExtractorChain())

    @cached_property
    def claim_extractor(self) -> ClaimExtractor:
        return ClaimExtractor(self.client, self.page_fetcher, self.prompts, self.router)

    @cached_property
    def url_canonicaliser(self) -> UrlCanonicaliser:
        """Shared so the pipeline and the search service agree on what "the same page"
        means. Two instances would not disagree today, but a rule added to one and not the
        other would deduplicate differently in two places — and the symptom would be a
        report citing one source twice."""
        return UrlCanonicaliser()

    @cached_property
    def search(self) -> SearchService:
        """Web search, with the repository injected so every query is recorded."""
        return SearchService(self._settings, repository=self.repository,
                             canonicaliser=self.url_canonicaliser)

    @cached_property
    def result_ranker(self) -> ResultRanker:
        return ResultRanker(self.client, self.prompts, self.router)

    # ── monitoring ──

    @cached_property
    def lms(self) -> LmsCommandRunner:
        return LmsCommandRunner(self._settings)

    @cached_property
    def router(self) -> ModelRouter:
        """Chooses which installed model runs a given kind of work.

        Built from the registry rather than a fixed list, so a model downloaded a minute
        ago is routed to without restarting anything.
        """
        return ModelRouter(self._settings, self.model_registry)

    @cached_property
    def model_registry(self) -> ModelRegistry:
        return ModelRegistry(self.lms)

    @cached_property
    def server_probe(self) -> ModelServerProbe:
        return ModelServerProbe(self._settings)

    @cached_property
    def transcript_locator(self) -> TranscriptLocator:
        """Shared by the two readers that consume Claude Code's transcripts.

        One instance so both agree on where the transcripts are; it holds no cache, so
        sharing it is about consistency rather than cost.
        """
        return TranscriptLocator(self._settings)

    @cached_property
    def usage_reader(self) -> ClaudeUsageReader:
        return ClaudeUsageReader(self._settings, self.transcript_locator)

    @cached_property
    def agent_scanner(self) -> AgentTranscriptScanner:
        """A singleton because its whole value is the state it accumulates.

        The scanner remembers how far into each transcript it has read. Two instances
        would each re-read every file from the start, which is precisely the cost the
        byte-offset tailing exists to avoid — so this must be one shared object, not a
        fresh one per request.
        """
        return AgentTranscriptScanner()

    @cached_property
    def agent_reader(self) -> AgentActivityReader:
        return AgentActivityReader(
            self._settings, self.transcript_locator, self.agent_scanner
        )

    @cached_property
    def monitor(self) -> SystemMonitor:
        return SystemMonitor(
            GpuProbe(),
            HostProbe(),
            self.model_registry,
            self.server_probe,
            self.usage_reader,
        )
