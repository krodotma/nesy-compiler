"""
ARK Performance & Safety

Phase 2.5 Implementation (P2-081 to P2-100)

Provides:
- Gate execution profiling
- Parallel gate execution
- Result caching with bloom filters
- Circuit breaker pattern
- Kill switch integration
- Spectral stability monitoring
- Rate limiting and resource guards
"""

from .profiler import GateProfiler, ProfileResult, PerformanceMetrics
from .cache import GateCache, BloomFilter
from .parallel import ParallelGateExecutor, AsyncCommitPipeline
from .circuit import CircuitBreaker, CircuitState
from .safety import KillSwitch, SpectralMonitor, AnomalyDetector, RateLimiter
from .telemetry import Telemetry, MetricCollector
