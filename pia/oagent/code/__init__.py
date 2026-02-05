#!/usr/bin/env python3
"""
Code Agent - Neural-Guided Code Generation and AST Manipulation

Part of OAGENT 300-Step Plan: Steps 51-100

This module provides:
- Bootstrap and lifecycle management (Step 51)
- Neural code proposal generation (Step 52)
- AST transformations for Python/TypeScript (Steps 53-55)
- Multi-file edit coordination (Step 56)
- PAIP isolation clone management (Step 57)
- Import dependency resolution (Step 58)
- Code style enforcement (Step 59)
- Incremental compilation (Step 60)
- Diff optimization (Step 61)
- Conflict resolution (Step 62)
- Semantic merging (Step 63)
- Rollback management (Step 64)
- Checkpoint system (Step 65)
- Transaction logging (Step 66)
- Undo/redo stack (Step 67)
- Workspace sync (Step 68)
- File watching (Step 69)
- Code orchestration (Step 70)
- Code generation (Step 71)
- Template engine (Step 72)
- Snippet manager (Step 73)
- Refactoring engine (Step 74)
- Code formatter (Step 75)
- Unified linter (Step 76)
- Type checker (Step 77)
- Documentation generator (Step 78)
- Code API (Step 79)
- Code CLI (Step 80)
- Plugin system (Step 81)
- Caching layer (Step 82)
- Metrics system (Step 83)
- Logging system (Step 84)
- Error handler (Step 85)
- Config manager (Step 86)
- Health check (Step 87)
- Rate limiter (Step 88)
- Batch processor (Step 89)
- Event emitter (Step 90)
- Security module (Step 91)
- Validation module (Step 92)
- Testing framework (Step 93)
- Documentation module (Step 94)
- Migration tools (Step 95)
- Backup system (Step 96)
- Telemetry module (Step 97)
- Versioning module (Step 98)
- Deprecation manager (Step 99)
- Final orchestrator (Step 100)

Protocol: DKIN v30, CITIZEN v2, PAIP v16
Bus Topics: a2a.code.*, code.*, paip.*
A2A Heartbeat: 300s interval, 900s timeout
FalkorDB: Port 6380
"""

from .bootstrap import CodeAgentBootstrap, CodeAgentConfig

# Steps 71-80 imports
from .generator import CodeGenerator, GeneratorConfig, GenerationRequest, GenerationResult
from .template import TemplateEngine, TemplateConfig, CodeTemplate, TemplateResult
from .snippet import SnippetManager, SnippetConfig, CodeSnippet, SnippetSearchResult
from .refactor import RefactoringEngine, RefactoringConfig, RefactoringOperation, RefactoringResult
from .formatter import CodeFormatter, FormatterConfig, FormatResult
from .linter import UnifiedLinter, LinterConfig, LintIssue, LintResult
from .typechecker import TypeChecker, TypeCheckerConfig, TypeIssue, TypeCheckResult
from .docs import DocumentationGenerator, DocConfig, DocResult
from .api import CodeAPI, APIConfig
from .cli import CodeCLI, CLIConfig

# Steps 81-90 imports
from .plugin import PluginSystem, PluginConfig, Plugin, PluginManager
from .cache import CachingLayer, CacheConfig, CacheEntry
from .metrics import MetricsSystem, MetricsConfig, Metric
from .logging import LoggingSystem, LoggingConfig, LogEntry
from .errorhandler import ErrorHandler, ErrorConfig, CodeError
from .configmanager import ConfigManager, ConfigSchema
from .healthcheck import HealthChecker, HealthCheckConfig, HealthStatus
from .ratelimiter import RateLimiter, RateLimiterConfig
from .batch import BatchProcessor, BatchConfig, BatchJob
from .events import EventEmitter, EventConfig, Event

# Steps 91-100 imports
from .security import SecurityModule, SecurityConfig, AuthToken, Permission, Role
from .validation import ValidationModule, ValidationConfig, ValidationResult
from .testing import TestingFramework, TestConfig, TestCase, TestResult
from .documentation import DocumentationModule, DocConfig as DocModuleConfig, APIDoc
from .migration import MigrationModule, MigrationConfig, Migration, MigrationResult
from .backup import BackupModule, BackupConfig, Backup, RestoreResult
from .telemetry import TelemetryModule, TelemetryConfig, TelemetryEvent, TelemetryMetric
from .versioning import VersioningModule, VersionConfig, APIVersion, VersioningStrategy
from .deprecation import DeprecationManager, DeprecationConfig, Deprecation, deprecated
from .final_orchestrator import FinalOrchestrator, OrchestratorConfig, AgentStatus

__all__ = [
    # Bootstrap (Step 51)
    "CodeAgentBootstrap",
    "CodeAgentConfig",
    # Generator (Step 71)
    "CodeGenerator",
    "GeneratorConfig",
    "GenerationRequest",
    "GenerationResult",
    # Template (Step 72)
    "TemplateEngine",
    "TemplateConfig",
    "CodeTemplate",
    "TemplateResult",
    # Snippet (Step 73)
    "SnippetManager",
    "SnippetConfig",
    "CodeSnippet",
    "SnippetSearchResult",
    # Refactor (Step 74)
    "RefactoringEngine",
    "RefactoringConfig",
    "RefactoringOperation",
    "RefactoringResult",
    # Formatter (Step 75)
    "CodeFormatter",
    "FormatterConfig",
    "FormatResult",
    # Linter (Step 76)
    "UnifiedLinter",
    "LinterConfig",
    "LintIssue",
    "LintResult",
    # TypeChecker (Step 77)
    "TypeChecker",
    "TypeCheckerConfig",
    "TypeIssue",
    "TypeCheckResult",
    # Documentation (Step 78)
    "DocumentationGenerator",
    "DocConfig",
    "DocResult",
    # API (Step 79)
    "CodeAPI",
    "APIConfig",
    # CLI (Step 80)
    "CodeCLI",
    "CLIConfig",
    # Plugin (Step 81)
    "PluginSystem",
    "PluginConfig",
    "Plugin",
    "PluginManager",
    # Cache (Step 82)
    "CachingLayer",
    "CacheConfig",
    "CacheEntry",
    # Metrics (Step 83)
    "MetricsSystem",
    "MetricsConfig",
    "Metric",
    # Logging (Step 84)
    "LoggingSystem",
    "LoggingConfig",
    "LogEntry",
    # Error Handler (Step 85)
    "ErrorHandler",
    "ErrorConfig",
    "CodeError",
    # Config Manager (Step 86)
    "ConfigManager",
    "ConfigSchema",
    # Health Check (Step 87)
    "HealthChecker",
    "HealthCheckConfig",
    "HealthStatus",
    # Rate Limiter (Step 88)
    "RateLimiter",
    "RateLimiterConfig",
    # Batch Processor (Step 89)
    "BatchProcessor",
    "BatchConfig",
    "BatchJob",
    # Event Emitter (Step 90)
    "EventEmitter",
    "EventConfig",
    "Event",
    # Security (Step 91)
    "SecurityModule",
    "SecurityConfig",
    "AuthToken",
    "Permission",
    "Role",
    # Validation (Step 92)
    "ValidationModule",
    "ValidationConfig",
    "ValidationResult",
    # Testing (Step 93)
    "TestingFramework",
    "TestConfig",
    "TestCase",
    "TestResult",
    # Documentation Module (Step 94)
    "DocumentationModule",
    "DocModuleConfig",
    "APIDoc",
    # Migration (Step 95)
    "MigrationModule",
    "MigrationConfig",
    "Migration",
    "MigrationResult",
    # Backup (Step 96)
    "BackupModule",
    "BackupConfig",
    "Backup",
    "RestoreResult",
    # Telemetry (Step 97)
    "TelemetryModule",
    "TelemetryConfig",
    "TelemetryEvent",
    "TelemetryMetric",
    # Versioning (Step 98)
    "VersioningModule",
    "VersionConfig",
    "APIVersion",
    "VersioningStrategy",
    # Deprecation (Step 99)
    "DeprecationManager",
    "DeprecationConfig",
    "Deprecation",
    "deprecated",
    # Final Orchestrator (Step 100)
    "FinalOrchestrator",
    "OrchestratorConfig",
    "AgentStatus",
]

__version__ = "1.0.0"
