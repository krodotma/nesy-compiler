#!/usr/bin/env python3
"""
Research Agent - Oagent Subagent 1 (Steps 1-50) - COMPLETE

The Research Agent handles codebase exploration, documentation retrieval,
knowledge graph population, and context optimization.

PBTSO Phases: SKILL, SEQUESTER, RESEARCH, DISTILL, PLAN, ITERATE, INTERFACE,
              EXTEND, MONITOR, PROTECT, OPTIMIZE, COORDINATE

Steps 21-30:
- Step 21: Query Executor - Execute planned research queries
- Step 22: Result Ranker - Rank and prioritize results
- Step 23: Answer Synthesizer - Generate coherent answers
- Step 24: Citation Generator - Track and format citations
- Step 25: Confidence Scorer - Score answer confidence
- Step 26: Feedback Integrator - Learn from feedback
- Step 27: Incremental Updater - Update indexes incrementally
- Step 28: Multi-Repo Manager - Handle multiple repositories
- Step 29: Research API - REST API for research queries
- Step 30: Research CLI - Complete CLI interface

Steps 31-40 (Infrastructure):
- Step 31: Plugin System - Extensible plugin architecture
- Step 32: Caching Layer - Multi-tier caching (memory, disk, distributed)
- Step 33: Metrics - Performance and usage metrics
- Step 34: Logging - Structured logging system
- Step 35: Error Handler - Comprehensive error handling
- Step 36: Config Manager - Configuration management
- Step 37: Health Check - Health monitoring
- Step 38: Rate Limiter - API rate limiting
- Step 39: Batch Processor - Batch query processing
- Step 40: Event Emitter - Event emission system

Steps 41-50 (Final Components):
- Step 41: Security Module - Authentication, authorization, audit
- Step 42: Validation - Input/output validation, sanitization
- Step 43: Testing Framework - Unit/integration tests, mocking
- Step 44: Documentation - API docs, auto-generation
- Step 45: Migration Tools - Data migration, versioning
- Step 46: Backup System - Backup/restore capabilities
- Step 47: Telemetry - Usage analytics, metrics
- Step 48: Versioning - API versioning system
- Step 49: Deprecation Manager - Deprecation lifecycle
- Step 50: Final Orchestrator - Complete agent orchestration

Bus Topics:
- a2a.research.bootstrap.start
- a2a.research.bootstrap.complete
- a2a.research.execute.start
- a2a.research.execute.complete
- a2a.research.rank.start
- a2a.research.rank.complete
- a2a.research.synthesize.start
- a2a.research.synthesize.complete
- a2a.research.cite.generate
- a2a.research.confidence.score
- a2a.research.feedback.receive
- a2a.research.update.start
- a2a.research.repo.register
- a2a.research.api.request
- a2a.research.cli.execute
- research.scan.start
- research.scan.progress
- research.scan.complete
- research.parser.register
- research.capability.register
- research.index.symbol
- research.query.symbols

Protocol: DKIN v30, PAIP v16, HOLON v2, CITIZEN v2
"""
from __future__ import annotations

__version__ = "0.2.0"
__author__ = "isomorphics"

from .bootstrap import ResearchAgentBootstrap, ResearchAgentConfig
from .scanner import CodebaseScanner

# Steps 21-25: Executor components
from .executor import (
    QueryExecutor,
    ExecutionConfig,
    ExecutedQuery,
    ResultRanker,
    RankerConfig,
    RankedResult,
    AnswerSynthesizer,
    SynthesizerConfig,
    SynthesizedAnswer,
    CitationGenerator,
    CitationConfig,
    Citation,
    ConfidenceScorer,
    ScorerConfig,
    ConfidenceScore,
)

# Steps 26-27: Feedback components
from .feedback import (
    FeedbackIntegrator,
    FeedbackConfig,
    Feedback,
    IncrementalUpdater,
    UpdaterConfig,
    UpdateEvent,
)

# Step 28: Multi-repo management
from .multi_repo import (
    MultiRepoManager,
    RepoConfig,
    Repository,
)

# Step 29: API
from .api import (
    ResearchAPI,
    APIConfig,
)

# Step 30: CLI
from .cli import (
    ResearchCLI,
)

# Steps 31-40: Infrastructure components
from .infra import (
    # Step 31: Plugin System
    PluginManager,
    Plugin,
    PluginConfig,
    PluginMetadata,
    PluginHook,
    # Step 32: Caching Layer
    TieredCache,
    CacheTier,
    TieredCacheConfig,
    # Step 33: Metrics
    MetricsCollector,
    Metric,
    MetricType,
    MetricsConfig,
    Timer,
    Counter,
    Gauge,
    Histogram,
    # Step 34: Logging
    StructuredLogger,
    LogConfig,
    LogLevel,
    LogEntry,
    # Step 35: Error Handler
    ErrorHandler,
    ResearchError,
    ErrorCode,
    ErrorContext,
    ErrorConfig,
    # Step 36: Config Manager
    ConfigManager,
    ConfigSource,
    ConfigValue,
    ConfigSchema,
    ConfigType,
    # Step 37: Health Check
    HealthChecker,
    HealthStatus,
    HealthCheck,
    HealthConfig,
    ComponentHealth,
    CheckType,
    # Step 38: Rate Limiter
    RateLimiter,
    RateLimitConfig,
    RateLimitResult,
    TokenBucket,
    SlidingWindow,
    # Step 39: Batch Processor
    BatchProcessor,
    BatchConfig,
    BatchJob,
    BatchResult,
    BatchStatus,
    # Step 40: Event Emitter
    EventEmitter,
    Event,
    EventConfig,
    EventHandler,
    EventFilter,
)

# Steps 41-50: Final components
from .final import (
    # Step 41: Security Module
    SecurityManager,
    SecurityConfig,
    AuthMethod,
    Permission,
    Role,
    Principal,
    AuthToken,
    AuthResult,
    AuthzResult,
    AuditEntry,
    # Step 42: Validation
    ValidationManager,
    ValidationConfig,
    ValidationType,
    SanitizationType,
    ValidationError,
    ValidationResult,
    FieldValidator,
    TypeValidator,
    RangeValidator,
    LengthValidator,
    PatternValidator,
    EnumValidator,
    CustomValidator,
    Schema,
    FieldSchema,
    # Step 43: Testing Framework
    TestRunner,
    TestSuite,
    TestCase,
    TestResult,
    TestSuiteResult,
    TestConfig,
    TestStatus,
    TestType,
    Assertions,
    Mock,
    patch,
    Fixture,
    # Step 44: Documentation
    DocumentationManager,
    DocConfig,
    DocFormat,
    DocType,
    ModuleDoc,
    ClassDoc,
    FunctionDoc,
    # Step 45: Migration Tools
    MigrationManager,
    MigrationConfig,
    Migration,
    MigrationContext,
    MigrationRecord,
    MigrationResult,
    MigrationStatus,
    DataTransformer,
    # Step 46: Backup System
    BackupManager,
    BackupConfig,
    BackupManifest,
    RestoreResult,
    VerificationResult,
    BackupType,
    BackupStatus,
    # Step 47: Telemetry
    TelemetryCollector,
    TelemetryConfig,
    TelemetryEvent,
    UsageReport,
    EventCategory,
    # Step 48: Versioning
    VersionManager,
    VersionConfig,
    SemanticVersion,
    VersionedEndpoint,
    VersionNegotiationResult,
    ChangelogEntry,
    # Step 49: Deprecation Manager
    DeprecationManager,
    DeprecationConfig,
    DeprecationNotice,
    DeprecationPhase,
    DeprecationType,
    # Step 50: Final Orchestrator
    ResearchAgentOrchestrator,
    OrchestratorConfig,
    AgentState,
    ComponentStatus,
    A2A_HEARTBEAT_INTERVAL,
    A2A_HEARTBEAT_TIMEOUT,
    FALKORDB_PORT,
)

__all__ = [
    # Core
    "ResearchAgentBootstrap",
    "ResearchAgentConfig",
    "CodebaseScanner",
    # Step 21: Query Executor
    "QueryExecutor",
    "ExecutionConfig",
    "ExecutedQuery",
    # Step 22: Result Ranker
    "ResultRanker",
    "RankerConfig",
    "RankedResult",
    # Step 23: Answer Synthesizer
    "AnswerSynthesizer",
    "SynthesizerConfig",
    "SynthesizedAnswer",
    # Step 24: Citation Generator
    "CitationGenerator",
    "CitationConfig",
    "Citation",
    # Step 25: Confidence Scorer
    "ConfidenceScorer",
    "ScorerConfig",
    "ConfidenceScore",
    # Step 26: Feedback Integrator
    "FeedbackIntegrator",
    "FeedbackConfig",
    "Feedback",
    # Step 27: Incremental Updater
    "IncrementalUpdater",
    "UpdaterConfig",
    "UpdateEvent",
    # Step 28: Multi-Repo Manager
    "MultiRepoManager",
    "RepoConfig",
    "Repository",
    # Step 29: Research API
    "ResearchAPI",
    "APIConfig",
    # Step 30: Research CLI
    "ResearchCLI",
    # Step 31: Plugin System
    "PluginManager",
    "Plugin",
    "PluginConfig",
    "PluginMetadata",
    "PluginHook",
    # Step 32: Caching Layer
    "TieredCache",
    "CacheTier",
    "TieredCacheConfig",
    # Step 33: Metrics
    "MetricsCollector",
    "Metric",
    "MetricType",
    "MetricsConfig",
    "Timer",
    "Counter",
    "Gauge",
    "Histogram",
    # Step 34: Logging
    "StructuredLogger",
    "LogConfig",
    "LogLevel",
    "LogEntry",
    # Step 35: Error Handler
    "ErrorHandler",
    "ResearchError",
    "ErrorCode",
    "ErrorContext",
    "ErrorConfig",
    # Step 36: Config Manager
    "ConfigManager",
    "ConfigSource",
    "ConfigValue",
    "ConfigSchema",
    "ConfigType",
    # Step 37: Health Check
    "HealthChecker",
    "HealthStatus",
    "HealthCheck",
    "HealthConfig",
    "ComponentHealth",
    "CheckType",
    # Step 38: Rate Limiter
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    "TokenBucket",
    "SlidingWindow",
    # Step 39: Batch Processor
    "BatchProcessor",
    "BatchConfig",
    "BatchJob",
    "BatchResult",
    "BatchStatus",
    # Step 40: Event Emitter
    "EventEmitter",
    "Event",
    "EventConfig",
    "EventHandler",
    "EventFilter",
    # Step 41: Security Module
    "SecurityManager",
    "SecurityConfig",
    "AuthMethod",
    "Permission",
    "Role",
    "Principal",
    "AuthToken",
    "AuthResult",
    "AuthzResult",
    "AuditEntry",
    # Step 42: Validation
    "ValidationManager",
    "ValidationConfig",
    "ValidationType",
    "SanitizationType",
    "ValidationError",
    "ValidationResult",
    "FieldValidator",
    "TypeValidator",
    "RangeValidator",
    "LengthValidator",
    "PatternValidator",
    "EnumValidator",
    "CustomValidator",
    "Schema",
    "FieldSchema",
    # Step 43: Testing Framework
    "TestRunner",
    "TestSuite",
    "TestCase",
    "TestResult",
    "TestSuiteResult",
    "TestConfig",
    "TestStatus",
    "TestType",
    "Assertions",
    "Mock",
    "patch",
    "Fixture",
    # Step 44: Documentation
    "DocumentationManager",
    "DocConfig",
    "DocFormat",
    "DocType",
    "ModuleDoc",
    "ClassDoc",
    "FunctionDoc",
    # Step 45: Migration Tools
    "MigrationManager",
    "MigrationConfig",
    "Migration",
    "MigrationContext",
    "MigrationRecord",
    "MigrationResult",
    "MigrationStatus",
    "DataTransformer",
    # Step 46: Backup System
    "BackupManager",
    "BackupConfig",
    "BackupManifest",
    "RestoreResult",
    "VerificationResult",
    "BackupType",
    "BackupStatus",
    # Step 47: Telemetry
    "TelemetryCollector",
    "TelemetryConfig",
    "TelemetryEvent",
    "UsageReport",
    "EventCategory",
    # Step 48: Versioning
    "VersionManager",
    "VersionConfig",
    "SemanticVersion",
    "VersionedEndpoint",
    "VersionNegotiationResult",
    "ChangelogEntry",
    # Step 49: Deprecation Manager
    "DeprecationManager",
    "DeprecationConfig",
    "DeprecationNotice",
    "DeprecationPhase",
    "DeprecationType",
    # Step 50: Final Orchestrator
    "ResearchAgentOrchestrator",
    "OrchestratorConfig",
    "AgentState",
    "ComponentStatus",
    "A2A_HEARTBEAT_INTERVAL",
    "A2A_HEARTBEAT_TIMEOUT",
    "FALKORDB_PORT",
]
