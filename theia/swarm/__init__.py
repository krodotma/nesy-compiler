"""
Theia Swarm Module — Multi-agent collaboration.

Provides:
- SwarmDispatcher: Prescient routing with failover
- TaskLifecycle: DKIN v18 compliant events (DEPRECATED)
- A2AProtocol: DKIN v29 compliant A2A handshake, heartbeat, codeword
- TimeoutHandler: A2A timeout with auto-reroute
- ErrorRateMonitor: Circuit breaker pattern
"""

from theia.swarm.dispatcher import (
    PROVIDER_PRIORITY,
    DispatchRequest,
    SwarmDispatcher,
    TaskLifecycle,  # Deprecated: use A2AProtocol
)

from theia.swarm.a2a_protocol import (
    PROTO_HEADER,
    CollabStatus,
    Collaboration,
    A2AProtocol,
)

from theia.swarm.timeout_handler import (
    PendingRequest,
    TimeoutHandler,
)

from theia.swarm.error_monitor import (
    CircuitState,
    CircuitBreaker,
    ErrorRateMonitor,
)

__all__ = [
    # DKIN v29 (Canonical)
    "PROTO_HEADER",
    "CollabStatus",
    "Collaboration",
    "A2AProtocol",
    # Dispatcher
    "PROVIDER_PRIORITY",
    "DispatchRequest",
    "SwarmDispatcher",
    "TaskLifecycle",
    # Timeout
    "PendingRequest",
    "TimeoutHandler",
    # Error Monitor
    "CircuitState",
    "CircuitBreaker",
    "ErrorRateMonitor",
]
