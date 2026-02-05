#!/usr/bin/env python3
"""
Review Infrastructure Module (Steps 181-190)

Infrastructure components for the Review Agent including:
- Plugin System (Step 181)
- Caching Layer (Step 182)
- Metrics (Step 183)
- Logging (Step 184)
- Error Handler (Step 185)
- Config Manager (Step 186)
- Health Check (Step 187)
- Rate Limiter (Step 188)
- Batch Processor (Step 189)
- Event Emitter (Step 190)

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from .plugin_system import (
    PluginManager,
    ReviewPlugin,
    PluginConfig,
    PluginInfo,
    PluginHook,
    PluginState,
)

from .caching_layer import (
    CacheManager,
    CacheEntry,
    CacheConfig,
    CacheTier,
    CacheStats,
)

from .metrics import (
    MetricsCollector,
    Metric,
    MetricConfig,
    MetricName,
    MetricUnit,
    MetricAggregation,
)

from .logging_system import (
    StructuredLogger,
    LogConfig,
    LogLevel,
    LogEntry,
    LogContext,
)

from .error_handler import (
    ErrorHandler,
    ReviewError,
    ErrorConfig,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
)

from .config_manager import (
    ConfigManager,
    ConfigSchema,
    ConfigSource,
    ConfigValue,
    ConfigValidation,
)

from .health_check import (
    HealthChecker,
    HealthConfig,
    HealthStatus,
    HealthCheck,
    CheckResult,
)

from .rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitResult,
    TokenBucket,
    RateLimitStrategy,
)

from .batch_processor import (
    BatchProcessor,
    BatchConfig,
    BatchResult,
    BatchItem,
    BatchState,
)

from .event_emitter import (
    EventEmitter,
    Event,
    EventConfig,
    EventSubscription,
    EventPriority,
)

__all__ = [
    # Plugin System
    "PluginManager",
    "ReviewPlugin",
    "PluginConfig",
    "PluginInfo",
    "PluginHook",
    "PluginState",
    # Caching Layer
    "CacheManager",
    "CacheEntry",
    "CacheConfig",
    "CacheTier",
    "CacheStats",
    # Metrics
    "MetricsCollector",
    "Metric",
    "MetricConfig",
    "MetricName",
    "MetricUnit",
    "MetricAggregation",
    # Logging
    "StructuredLogger",
    "LogConfig",
    "LogLevel",
    "LogEntry",
    "LogContext",
    # Error Handler
    "ErrorHandler",
    "ReviewError",
    "ErrorConfig",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    # Config Manager
    "ConfigManager",
    "ConfigSchema",
    "ConfigSource",
    "ConfigValue",
    "ConfigValidation",
    # Health Check
    "HealthChecker",
    "HealthConfig",
    "HealthStatus",
    "HealthCheck",
    "CheckResult",
    # Rate Limiter
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    "TokenBucket",
    "RateLimitStrategy",
    # Batch Processor
    "BatchProcessor",
    "BatchConfig",
    "BatchResult",
    "BatchItem",
    "BatchState",
    # Event Emitter
    "EventEmitter",
    "Event",
    "EventConfig",
    "EventSubscription",
    "EventPriority",
]
