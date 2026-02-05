#!/usr/bin/env python3
"""
Infrastructure Components for Research Agent (Steps 31-40)

This module provides core infrastructure components:
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

Bus Topics:
- a2a.research.plugin.load
- a2a.research.plugin.unload
- a2a.research.cache.tier_hit
- a2a.research.cache.tier_miss
- a2a.research.metrics.collect
- a2a.research.log.entry
- a2a.research.error.handle
- a2a.research.config.change
- a2a.research.health.check
- a2a.research.rate_limit.exceeded
- a2a.research.batch.process
- a2a.research.event.emit

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

# Step 31: Plugin System
from .plugin_system import (
    PluginManager,
    Plugin,
    PluginConfig,
    PluginMetadata,
    PluginHook,
)

# Step 32: Caching Layer
from .caching_layer import (
    TieredCache,
    CacheTier,
    TieredCacheConfig,
    CacheStats as TierCacheStats,
)

# Step 33: Metrics
from .metrics import (
    MetricsCollector,
    Metric,
    MetricType,
    MetricsConfig,
    Timer,
    Counter,
    Gauge,
    Histogram,
)

# Step 34: Logging
from .logging import (
    StructuredLogger,
    LogConfig,
    LogLevel,
    LogEntry,
)

# Step 35: Error Handler
from .error_handler import (
    ErrorHandler,
    ResearchError,
    ErrorCode,
    ErrorContext,
    ErrorConfig,
)

# Step 36: Config Manager
from .config_manager import (
    ConfigManager,
    ConfigSource,
    ConfigValue,
    ConfigSchema,
    ConfigType,
)

# Step 37: Health Check
from .health_check import (
    HealthChecker,
    HealthStatus,
    HealthCheck,
    HealthConfig,
    ComponentHealth,
    CheckType,
)

# Step 38: Rate Limiter
from .rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitResult,
    TokenBucket,
    SlidingWindow,
)

# Step 39: Batch Processor
from .batch_processor import (
    BatchProcessor,
    BatchConfig,
    BatchJob,
    BatchResult,
    BatchStatus,
)

# Step 40: Event Emitter
from .event_emitter import (
    EventEmitter,
    Event,
    EventConfig,
    EventHandler,
    EventFilter,
)

__all__ = [
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
    "TierCacheStats",
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
]
