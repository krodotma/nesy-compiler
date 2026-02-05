#!/usr/bin/env python3
"""
rate_limiter.py - API Rate Limiting (Step 88)

PBTSO Phase: VERIFY

Provides:
- Multiple rate limiting algorithms
- Per-client/per-operation limits
- Burst allowance
- Rate limit headers
- Distributed rate limiting support

Bus Topics:
- code.ratelimit.exceeded
- code.ratelimit.allow
- code.ratelimit.stats

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class RateLimitPolicy(Enum):
    """Rate limiting policy."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_size: int = 20
    policy: RateLimitPolicy = RateLimitPolicy.TOKEN_BUCKET
    enable_headers: bool = True
    enable_bus_events: bool = True
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_per_second": self.requests_per_second,
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "burst_size": self.burst_size,
            "policy": self.policy.value,
        }


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    limit: int
    reset_at: float
    retry_after: Optional[float] = None
    client_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "limit": self.limit,
            "reset_at": self.reset_at,
            "retry_after": self.retry_after,
            "client_id": self.client_id,
        }

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP rate limit headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Rate Limiting Algorithms
# =============================================================================

class RateLimitAlgorithm(ABC):
    """Abstract base class for rate limiting algorithms."""

    @abstractmethod
    def allow(self, client_id: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed."""
        pass

    @abstractmethod
    def get_stats(self, client_id: str) -> Dict[str, Any]:
        """Get statistics for a client."""
        pass

    @abstractmethod
    def reset(self, client_id: str) -> None:
        """Reset rate limit for a client."""
        pass


class TokenBucket(RateLimitAlgorithm):
    """
    Token bucket rate limiting.

    Tokens are added at a fixed rate, requests consume tokens.
    Allows bursting up to bucket capacity.
    """

    def __init__(
        self,
        rate: float = 10.0,  # tokens per second
        capacity: int = 20,   # bucket capacity
    ):
        self.rate = rate
        self.capacity = capacity
        self._buckets: Dict[str, Tuple[float, float]] = {}  # client -> (tokens, last_update)
        self._lock = Lock()

    def allow(self, client_id: str, cost: int = 1) -> RateLimitResult:
        now = time.time()

        with self._lock:
            if client_id not in self._buckets:
                self._buckets[client_id] = (float(self.capacity), now)

            tokens, last_update = self._buckets[client_id]

            # Add tokens based on time elapsed
            elapsed = now - last_update
            tokens = min(self.capacity, tokens + elapsed * self.rate)

            # Check if allowed
            if tokens >= cost:
                tokens -= cost
                self._buckets[client_id] = (tokens, now)

                return RateLimitResult(
                    allowed=True,
                    remaining=int(tokens),
                    limit=self.capacity,
                    reset_at=now + (self.capacity - tokens) / self.rate,
                    client_id=client_id,
                )
            else:
                # Calculate wait time
                wait_time = (cost - tokens) / self.rate
                self._buckets[client_id] = (tokens, now)

                return RateLimitResult(
                    allowed=False,
                    remaining=int(tokens),
                    limit=self.capacity,
                    reset_at=now + wait_time,
                    retry_after=wait_time,
                    client_id=client_id,
                )

    def get_stats(self, client_id: str) -> Dict[str, Any]:
        with self._lock:
            if client_id not in self._buckets:
                return {"tokens": self.capacity, "capacity": self.capacity}
            tokens, last_update = self._buckets[client_id]
            return {
                "tokens": tokens,
                "capacity": self.capacity,
                "rate": self.rate,
                "last_update": last_update,
            }

    def reset(self, client_id: str) -> None:
        with self._lock:
            if client_id in self._buckets:
                del self._buckets[client_id]


class SlidingWindow(RateLimitAlgorithm):
    """
    Sliding window rate limiting.

    Counts requests in a sliding time window.
    """

    def __init__(
        self,
        limit: int = 100,
        window_s: int = 60,
    ):
        self.limit = limit
        self.window_s = window_s
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

    def allow(self, client_id: str, cost: int = 1) -> RateLimitResult:
        now = time.time()
        cutoff = now - self.window_s

        with self._lock:
            # Remove old requests
            self._requests[client_id] = [
                ts for ts in self._requests[client_id] if ts > cutoff
            ]

            current_count = len(self._requests[client_id])
            remaining = self.limit - current_count

            if current_count + cost <= self.limit:
                # Add new requests
                for _ in range(cost):
                    self._requests[client_id].append(now)

                return RateLimitResult(
                    allowed=True,
                    remaining=max(0, remaining - cost),
                    limit=self.limit,
                    reset_at=now + self.window_s,
                    client_id=client_id,
                )
            else:
                # Find when oldest request will expire
                oldest = min(self._requests[client_id]) if self._requests[client_id] else now
                retry_after = oldest + self.window_s - now

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.limit,
                    reset_at=oldest + self.window_s,
                    retry_after=max(0, retry_after),
                    client_id=client_id,
                )

    def get_stats(self, client_id: str) -> Dict[str, Any]:
        now = time.time()
        cutoff = now - self.window_s

        with self._lock:
            requests = [ts for ts in self._requests[client_id] if ts > cutoff]
            return {
                "current_count": len(requests),
                "limit": self.limit,
                "window_s": self.window_s,
            }

    def reset(self, client_id: str) -> None:
        with self._lock:
            if client_id in self._requests:
                del self._requests[client_id]


class LeakyBucket(RateLimitAlgorithm):
    """
    Leaky bucket rate limiting.

    Requests are processed at a fixed rate.
    Excess requests are queued or rejected.
    """

    def __init__(
        self,
        rate: float = 10.0,    # requests per second
        capacity: int = 20,     # queue capacity
    ):
        self.rate = rate
        self.capacity = capacity
        self._queues: Dict[str, Tuple[int, float]] = {}  # client -> (queue_size, last_leak)
        self._lock = Lock()

    def allow(self, client_id: str, cost: int = 1) -> RateLimitResult:
        now = time.time()

        with self._lock:
            if client_id not in self._queues:
                self._queues[client_id] = (0, now)

            queue_size, last_leak = self._queues[client_id]

            # Leak requests based on time elapsed
            elapsed = now - last_leak
            leaked = int(elapsed * self.rate)
            queue_size = max(0, queue_size - leaked)

            # Check if can add to queue
            if queue_size + cost <= self.capacity:
                queue_size += cost
                self._queues[client_id] = (queue_size, now)

                return RateLimitResult(
                    allowed=True,
                    remaining=self.capacity - queue_size,
                    limit=self.capacity,
                    reset_at=now + queue_size / self.rate,
                    client_id=client_id,
                )
            else:
                # Queue full
                wait_time = (queue_size + cost - self.capacity) / self.rate
                self._queues[client_id] = (queue_size, now)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.capacity,
                    reset_at=now + wait_time,
                    retry_after=wait_time,
                    client_id=client_id,
                )

    def get_stats(self, client_id: str) -> Dict[str, Any]:
        with self._lock:
            if client_id not in self._queues:
                return {"queue_size": 0, "capacity": self.capacity}
            queue_size, last_leak = self._queues[client_id]
            return {
                "queue_size": queue_size,
                "capacity": self.capacity,
                "rate": self.rate,
                "last_leak": last_leak,
            }

    def reset(self, client_id: str) -> None:
        with self._lock:
            if client_id in self._queues:
                del self._queues[client_id]


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    API rate limiting system.

    PBTSO Phase: VERIFY

    Features:
    - Multiple rate limiting algorithms
    - Per-client limits
    - Per-operation limits
    - Burst handling
    - Rate limit headers

    Usage:
        limiter = RateLimiter(config)
        result = limiter.check("client-123")
        if result.allowed:
            process_request()
        else:
            return_429(result)
    """

    BUS_TOPICS = {
        "exceeded": "code.ratelimit.exceeded",
        "allow": "code.ratelimit.allow",
        "stats": "code.ratelimit.stats",
    }

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or RateLimitConfig()
        self.bus = bus or LockedAgentBus()

        # Create algorithm based on policy
        self._algorithm = self._create_algorithm()

        # Per-operation limiters
        self._operation_limiters: Dict[str, RateLimitAlgorithm] = {}

        # Statistics
        self._total_requests = 0
        self._allowed_requests = 0
        self._denied_requests = 0
        self._lock = Lock()

    def _create_algorithm(self) -> RateLimitAlgorithm:
        """Create rate limit algorithm based on policy."""
        if self.config.policy == RateLimitPolicy.TOKEN_BUCKET:
            return TokenBucket(
                rate=self.config.requests_per_second,
                capacity=self.config.burst_size,
            )
        elif self.config.policy == RateLimitPolicy.SLIDING_WINDOW:
            return SlidingWindow(
                limit=self.config.requests_per_minute,
                window_s=60,
            )
        elif self.config.policy == RateLimitPolicy.LEAKY_BUCKET:
            return LeakyBucket(
                rate=self.config.requests_per_second,
                capacity=self.config.burst_size,
            )
        else:
            return TokenBucket(
                rate=self.config.requests_per_second,
                capacity=self.config.burst_size,
            )

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def check(
        self,
        client_id: str,
        cost: int = 1,
        operation: Optional[str] = None,
    ) -> RateLimitResult:
        """
        Check if request is allowed.

        Args:
            client_id: Client identifier
            cost: Request cost (for weighted limiting)
            operation: Optional operation name for per-operation limits

        Returns:
            RateLimitResult
        """
        # Check per-operation limit first
        if operation and operation in self._operation_limiters:
            op_result = self._operation_limiters[operation].allow(client_id, cost)
            if not op_result.allowed:
                self._record_denied(client_id, operation)
                return op_result

        # Check global limit
        result = self._algorithm.allow(client_id, cost)

        with self._lock:
            self._total_requests += 1
            if result.allowed:
                self._allowed_requests += 1
            else:
                self._denied_requests += 1

        # Emit bus event
        if self.config.enable_bus_events:
            if result.allowed:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["allow"],
                    "kind": "ratelimit",
                    "actor": "rate-limiter",
                    "data": {
                        "client_id": client_id,
                        "remaining": result.remaining,
                    },
                })
            else:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["exceeded"],
                    "kind": "ratelimit",
                    "level": "warning",
                    "actor": "rate-limiter",
                    "data": {
                        "client_id": client_id,
                        "retry_after": result.retry_after,
                    },
                })

        return result

    def _record_denied(self, client_id: str, operation: str) -> None:
        """Record a denied request."""
        with self._lock:
            self._total_requests += 1
            self._denied_requests += 1

    def allow(
        self,
        client_id: str,
        cost: int = 1,
        operation: Optional[str] = None,
    ) -> bool:
        """Simple check returning just bool."""
        return self.check(client_id, cost, operation).allowed

    # =========================================================================
    # Operation Limits
    # =========================================================================

    def set_operation_limit(
        self,
        operation: str,
        rate: float,
        capacity: int,
        policy: RateLimitPolicy = RateLimitPolicy.TOKEN_BUCKET,
    ) -> None:
        """Set rate limit for a specific operation."""
        if policy == RateLimitPolicy.TOKEN_BUCKET:
            self._operation_limiters[operation] = TokenBucket(rate, capacity)
        elif policy == RateLimitPolicy.SLIDING_WINDOW:
            self._operation_limiters[operation] = SlidingWindow(capacity, int(capacity / rate))
        else:
            self._operation_limiters[operation] = TokenBucket(rate, capacity)

    def remove_operation_limit(self, operation: str) -> bool:
        """Remove operation limit."""
        if operation in self._operation_limiters:
            del self._operation_limiters[operation]
            return True
        return False

    # =========================================================================
    # Management
    # =========================================================================

    def reset(self, client_id: str) -> None:
        """Reset rate limit for a client."""
        self._algorithm.reset(client_id)
        for limiter in self._operation_limiters.values():
            limiter.reset(client_id)

    def get_stats(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            stats = {
                "total_requests": self._total_requests,
                "allowed_requests": self._allowed_requests,
                "denied_requests": self._denied_requests,
                "deny_rate": self._denied_requests / self._total_requests if self._total_requests > 0 else 0,
                "config": self.config.to_dict(),
            }

        if client_id:
            stats["client"] = self._algorithm.get_stats(client_id)

        return stats


# =============================================================================
# Decorator
# =============================================================================

T = TypeVar("T")


def rate_limited(
    limiter: RateLimiter,
    client_id_func: Optional[Callable[..., str]] = None,
    cost: int = 1,
    operation: Optional[str] = None,
) -> Callable:
    """
    Decorator for rate limiting functions.

    Usage:
        @rate_limited(limiter, client_id_func=lambda x: x.client_id)
        def handle_request(request):
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get client ID
            if client_id_func:
                client_id = client_id_func(*args, **kwargs)
            else:
                client_id = "default"

            result = limiter.check(client_id, cost, operation)
            if not result.allowed:
                raise RateLimitExceeded(result)

            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            if client_id_func:
                client_id = client_id_func(*args, **kwargs)
            else:
                client_id = "default"

            result = limiter.check(client_id, cost, operation)
            if not result.allowed:
                raise RateLimitExceeded(result)

            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, result: RateLimitResult):
        self.result = result
        super().__init__(f"Rate limit exceeded. Retry after {result.retry_after}s")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Rate Limiter."""
    import argparse

    parser = argparse.ArgumentParser(description="Rate Limiter (Step 88)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check command
    check_parser = subparsers.add_parser("check", help="Check rate limit")
    check_parser.add_argument("client_id", help="Client ID")
    check_parser.add_argument("--cost", "-c", type=int, default=1, help="Request cost")
    check_parser.add_argument("--json", action="store_true", help="JSON output")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--client", "-c", help="Client ID")

    # reset command
    reset_parser = subparsers.add_parser("reset", help="Reset client limit")
    reset_parser.add_argument("client_id", help="Client ID")

    # demo command
    subparsers.add_parser("demo", help="Run rate limiting demo")

    args = parser.parse_args()
    limiter = RateLimiter()

    if args.command == "check":
        result = limiter.check(args.client_id, args.cost)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "ALLOWED" if result.allowed else "DENIED"
            print(f"{status}: {args.client_id}")
            print(f"  Remaining: {result.remaining}/{result.limit}")
            if result.retry_after:
                print(f"  Retry after: {result.retry_after:.2f}s")
        return 0 if result.allowed else 1

    elif args.command == "stats":
        stats = limiter.get_stats(args.client)
        print(json.dumps(stats, indent=2))
        return 0

    elif args.command == "reset":
        limiter.reset(args.client_id)
        print(f"Reset rate limit for: {args.client_id}")
        return 0

    elif args.command == "demo":
        print("Running rate limiter demo...")
        print(f"Config: {limiter.config.requests_per_second} req/s, burst: {limiter.config.burst_size}\n")

        client = "demo-client"

        # Burst of requests
        print("Sending burst of 25 requests:")
        for i in range(25):
            result = limiter.check(client)
            status = "OK" if result.allowed else "DENIED"
            print(f"  Request {i + 1}: {status} (remaining: {result.remaining})")

        # Wait and try again
        print("\nWaiting 1 second...")
        time.sleep(1)

        result = limiter.check(client)
        status = "OK" if result.allowed else "DENIED"
        print(f"After wait: {status} (remaining: {result.remaining})")

        print("\nStats:")
        print(json.dumps(limiter.get_stats(client), indent=2))

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
