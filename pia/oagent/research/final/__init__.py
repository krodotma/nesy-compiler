#!/usr/bin/env python3
"""
Final Components for Research Agent (Steps 41-50)

This module provides the final components completing the Research Agent:

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
- a2a.research.security.*
- a2a.research.validation.*
- a2a.research.test.*
- a2a.research.docs.*
- a2a.research.migration.*
- a2a.research.backup.*
- a2a.research.telemetry.*
- a2a.research.version.*
- a2a.research.deprecation.*
- a2a.research.orchestrator.*
- a2a.research.heartbeat

Protocol: DKIN v30, PAIP v16, HOLON v2, CITIZEN v2

A2A Heartbeat: 300s interval, 900s timeout
FalkorDB: port 6380
"""
from __future__ import annotations

# Step 41: Security Module
from .security_module import (
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
    APIKeyAuthProvider,
    JWTAuthProvider,
)

# Step 42: Validation
from .validation import (
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
    EMAIL_VALIDATOR,
    URL_VALIDATOR,
    UUID_VALIDATOR,
    SAFE_PATH_VALIDATOR,
)

# Step 43: Testing Framework
from .testing_framework import (
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
    SimpleFixture,
    TempDirFixture,
    MockBusFixture,
    MockBus,
    assert_,
)

# Step 44: Documentation
from .documentation import (
    DocumentationManager,
    DocConfig,
    DocFormat,
    DocType,
    ModuleDoc,
    ClassDoc,
    FunctionDoc,
    Parameter,
    ReturnValue,
    Example,
    APIEndpoint,
    DocstringParser,
    ModuleExtractor,
    MarkdownGenerator,
    HTMLGenerator,
)

# Step 45: Migration Tools
from .migration_tools import (
    MigrationManager,
    MigrationConfig,
    Migration,
    MigrationContext,
    MigrationRecord,
    MigrationResult,
    MigrationStatus,
    MigrationDirection,
    DataTransformation,
    DataTransformer,
)

# Step 46: Backup System
from .backup_system import (
    BackupManager,
    BackupConfig,
    BackupManifest,
    RestoreResult,
    VerificationResult,
    BackupType,
    BackupStatus,
    CompressionType,
)

# Step 47: Telemetry
from .telemetry import (
    TelemetryCollector,
    TelemetryConfig,
    TelemetryEvent,
    MetricValue,
    AggregatedMetric,
    UsageReport,
    EventCategory,
    MetricType as TelemetryMetricType,
)

# Step 48: Versioning
from .versioning import (
    VersionManager,
    VersionConfig,
    SemanticVersion,
    VersionedEndpoint,
    VersionNegotiationResult,
    ChangelogEntry,
    VersionScheme,
    CompatibilityLevel,
)

# Step 49: Deprecation Manager
from .deprecation_manager import (
    DeprecationManager,
    DeprecationConfig,
    DeprecationNotice,
    DeprecationUsage,
    DeprecationReport,
    DeprecationPhase,
    DeprecationType,
    DeprecationError,
)

# Step 50: Final Orchestrator
from .final_orchestrator import (
    ResearchAgentOrchestrator,
    OrchestratorConfig,
    AgentState,
    ComponentStatus,
    ComponentInfo,
    A2A_HEARTBEAT_INTERVAL,
    A2A_HEARTBEAT_TIMEOUT,
    FALKORDB_PORT,
)

__all__ = [
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
    "APIKeyAuthProvider",
    "JWTAuthProvider",
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
    "EMAIL_VALIDATOR",
    "URL_VALIDATOR",
    "UUID_VALIDATOR",
    "SAFE_PATH_VALIDATOR",
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
    "SimpleFixture",
    "TempDirFixture",
    "MockBusFixture",
    "MockBus",
    "assert_",
    # Step 44: Documentation
    "DocumentationManager",
    "DocConfig",
    "DocFormat",
    "DocType",
    "ModuleDoc",
    "ClassDoc",
    "FunctionDoc",
    "Parameter",
    "ReturnValue",
    "Example",
    "APIEndpoint",
    "DocstringParser",
    "ModuleExtractor",
    "MarkdownGenerator",
    "HTMLGenerator",
    # Step 45: Migration Tools
    "MigrationManager",
    "MigrationConfig",
    "Migration",
    "MigrationContext",
    "MigrationRecord",
    "MigrationResult",
    "MigrationStatus",
    "MigrationDirection",
    "DataTransformation",
    "DataTransformer",
    # Step 46: Backup System
    "BackupManager",
    "BackupConfig",
    "BackupManifest",
    "RestoreResult",
    "VerificationResult",
    "BackupType",
    "BackupStatus",
    "CompressionType",
    # Step 47: Telemetry
    "TelemetryCollector",
    "TelemetryConfig",
    "TelemetryEvent",
    "MetricValue",
    "AggregatedMetric",
    "UsageReport",
    "EventCategory",
    "TelemetryMetricType",
    # Step 48: Versioning
    "VersionManager",
    "VersionConfig",
    "SemanticVersion",
    "VersionedEndpoint",
    "VersionNegotiationResult",
    "ChangelogEntry",
    "VersionScheme",
    "CompatibilityLevel",
    # Step 49: Deprecation Manager
    "DeprecationManager",
    "DeprecationConfig",
    "DeprecationNotice",
    "DeprecationUsage",
    "DeprecationReport",
    "DeprecationPhase",
    "DeprecationType",
    "DeprecationError",
    # Step 50: Final Orchestrator
    "ResearchAgentOrchestrator",
    "OrchestratorConfig",
    "AgentState",
    "ComponentStatus",
    "ComponentInfo",
    "A2A_HEARTBEAT_INTERVAL",
    "A2A_HEARTBEAT_TIMEOUT",
    "FALKORDB_PORT",
]
