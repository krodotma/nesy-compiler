#!/usr/bin/env python3
"""
Test Agent - OAGENT Subagent 3 (Steps 101-150)

Automated test generation, mutation testing, and coverage analysis.

PBTSO Phases:
- SKILL: Core test generation capabilities
- SEQUESTER: Test isolation and sandboxing
- TEST: Test execution and result collection
- VERIFY: Coverage and mutation verification
- OBSERVE: Reporting and monitoring
- DISTRIBUTE: Parallel execution

A2A Topics:
- a2a.test.bootstrap.start
- a2a.test.bootstrap.complete
- test.unit.generate / test.unit.generated
- test.integration.generate
- test.e2e.generate
- test.property.generate
- test.run.start / test.run.complete
- test.pytest.run / test.vitest.run
- test.coverage.analyze / test.coverage.report
- test.report.generate / test.report.complete
- test.dashboard.update / test.dashboard.status
- test.history.record / test.history.query
- test.compare.request / test.compare.result
- test.notify.send / test.notify.sent
- test.schedule.add / test.schedule.run
- test.parallel.start / test.parallel.complete
- test.cache.hit / test.cache.miss
- test.api.request / test.api.response
- test.cli.command / test.cli.result
- test.plugin.load / test.plugin.hook
- test.caching.hit / test.caching.store
- test.metrics.record / test.metrics.report
- test.log.entry / test.log.error
- test.error.occurred / test.error.recovered
- test.config.load / test.config.change
- test.health.check / test.health.status
- test.ratelimit.allowed / test.ratelimit.denied
- test.batch.start / test.batch.complete
- test.event.emit / test.event.subscribe
- telemetry.test.coverage

Steps 121-130:
- Step 121: Test Report Generator
- Step 122: Test Dashboard
- Step 123: Test History Tracker
- Step 124: Test Comparison
- Step 125: Test Notification
- Step 126: Test Scheduling
- Step 127: Test Parallelizer
- Step 128: Test Cache
- Step 129: Test API
- Step 130: Test CLI

Steps 131-140:
- Step 131: Test Plugin System
- Step 132: Test Caching Layer
- Step 133: Test Metrics
- Step 134: Test Logging
- Step 135: Test Error Handler
- Step 136: Test Config Manager
- Step 137: Test Health Check
- Step 138: Test Rate Limiter
- Step 139: Test Batch Processor
- Step 140: Test Event Emitter
"""

from .bootstrap import (
    TestAgentConfig,
    TestAgentBootstrap,
    PBTSOPhase,
)

# Step 121: Test Report Generator
from .report import (
    TestReportGenerator,
    ReportConfig,
    ReportResult,
    ReportFormat,
)

# Step 122: Test Dashboard
from .dashboard import (
    TestDashboard,
    DashboardConfig,
    DashboardState,
    DashboardMetrics,
)

# Step 123: Test History Tracker
from .history import (
    TestHistoryTracker,
    HistoryConfig,
    HistoryRecord,
    HistoryQuery,
)

# Step 124: Test Comparison
from .compare import (
    TestComparator,
    CompareConfig,
    CompareResult,
    TestDiff,
)

# Step 125: Test Notification
from .notify import (
    TestNotifier,
    NotifyConfig,
    NotifyResult,
    NotificationChannel,
    AlertRule,
)

# Step 126: Test Scheduling
from .schedule import (
    TestScheduler,
    ScheduleConfig,
    ScheduledJob,
    ScheduleFrequency,
)

# Step 127: Test Parallelizer
from .parallel import (
    TestParallelizer,
    ParallelConfig,
    ParallelResult,
    PartitionStrategy,
)

# Step 128: Test Cache
from .cache import (
    TestCache,
    CacheConfig,
    CacheEntry,
    CacheStats,
)

# Step 129: Test API
from .api import (
    TestAPI,
    APIConfig,
    APIResponse,
)

# Step 130: Test CLI
from .cli import (
    TestCLI,
    CLIConfig,
    main as cli_main,
)

# Step 131: Test Plugin System
from .plugin import (
    TestPluginManager,
    PluginConfig,
    Plugin,
    PluginHook,
    PluginContext,
    PluginResult,
)

# Step 132: Test Caching Layer
from .caching import (
    TestCachingLayer,
    CachingConfig,
    CacheTier,
)

# Step 133: Test Metrics
from .metrics import (
    TestMetrics,
    MetricsConfig,
    MetricType,
    MetricValue,
    MetricsReport,
)

# Step 134: Test Logging
from .logging import (
    TestLogger,
    LogConfig,
    LogLevel,
    LogEntry,
)

# Step 135: Test Error Handler
from .errors import (
    TestErrorHandler,
    ErrorConfig,
    ErrorCategory,
    ErrorSeverity,
    ErrorInfo,
    RecoveryStrategy,
)

# Step 136: Test Config Manager
from .config import (
    TestConfigManager,
    ConfigSource,
    ConfigSchema,
    ConfigValue,
)

# Step 137: Test Health Check
from .health import (
    TestHealthCheck,
    HealthConfig,
    HealthStatus,
    HealthComponent,
    HealthReport,
)

# Step 138: Test Rate Limiter
from .ratelimit import (
    TestRateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    RateLimitResult,
)

# Step 139: Test Batch Processor
from .batch import (
    TestBatchProcessor,
    BatchConfig,
    BatchJob,
    BatchStatus,
    BatchResult,
)

# Step 140: Test Event Emitter
from .events import (
    TestEventEmitter,
    EventConfig,
    EventType,
    Event,
    EventSubscription,
)

__all__ = [
    # Bootstrap (Steps 101-110)
    "TestAgentConfig",
    "TestAgentBootstrap",
    "PBTSOPhase",
    # Step 121: Report
    "TestReportGenerator",
    "ReportConfig",
    "ReportResult",
    "ReportFormat",
    # Step 122: Dashboard
    "TestDashboard",
    "DashboardConfig",
    "DashboardState",
    "DashboardMetrics",
    # Step 123: History
    "TestHistoryTracker",
    "HistoryConfig",
    "HistoryRecord",
    "HistoryQuery",
    # Step 124: Compare
    "TestComparator",
    "CompareConfig",
    "CompareResult",
    "TestDiff",
    # Step 125: Notify
    "TestNotifier",
    "NotifyConfig",
    "NotifyResult",
    "NotificationChannel",
    "AlertRule",
    # Step 126: Schedule
    "TestScheduler",
    "ScheduleConfig",
    "ScheduledJob",
    "ScheduleFrequency",
    # Step 127: Parallel
    "TestParallelizer",
    "ParallelConfig",
    "ParallelResult",
    "PartitionStrategy",
    # Step 128: Cache
    "TestCache",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    # Step 129: API
    "TestAPI",
    "APIConfig",
    "APIResponse",
    # Step 130: CLI
    "TestCLI",
    "CLIConfig",
    "cli_main",
    # Step 131: Plugin System
    "TestPluginManager",
    "PluginConfig",
    "Plugin",
    "PluginHook",
    "PluginContext",
    "PluginResult",
    # Step 132: Caching Layer
    "TestCachingLayer",
    "CachingConfig",
    "CacheTier",
    # Step 133: Metrics
    "TestMetrics",
    "MetricsConfig",
    "MetricType",
    "MetricValue",
    "MetricsReport",
    # Step 134: Logging
    "TestLogger",
    "LogConfig",
    "LogLevel",
    "LogEntry",
    # Step 135: Error Handler
    "TestErrorHandler",
    "ErrorConfig",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorInfo",
    "RecoveryStrategy",
    # Step 136: Config Manager
    "TestConfigManager",
    "ConfigSource",
    "ConfigSchema",
    "ConfigValue",
    # Step 137: Health Check
    "TestHealthCheck",
    "HealthConfig",
    "HealthStatus",
    "HealthComponent",
    "HealthReport",
    # Step 138: Rate Limiter
    "TestRateLimiter",
    "RateLimitConfig",
    "RateLimitStrategy",
    "RateLimitResult",
    # Step 139: Batch Processor
    "TestBatchProcessor",
    "BatchConfig",
    "BatchJob",
    "BatchStatus",
    "BatchResult",
    # Step 140: Event Emitter
    "TestEventEmitter",
    "EventConfig",
    "EventType",
    "Event",
    "EventSubscription",
]

__version__ = "0.3.0"
__step_range__ = "101-140"
