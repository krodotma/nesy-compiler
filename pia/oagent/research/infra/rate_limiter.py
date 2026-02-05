#!/usr/bin/env python3
"""
rate_limiter.py - API Rate Limiting (Step 38)

Rate limiting for Research Agent API and operations.
Supports token bucket, sliding window, and fixed window algorithms.

PBTSO Phase: PROTECT

Bus Topics:
- a2a.research.rate_limit.exceeded
- a2a.research.rate_limit.reset
- research.rate_limit.quota

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""

    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    burst_size: int = 20
    window_size_seconds: int = 60
    enable_quota: bool = False
    quota_per_hour: int = 1000
    emit_to_bus: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None
    limit: int = 0
    client_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "retry_after": self.retry_after,
            "limit": self.limit,
            "client_id": self.client_id,
        }

    def headers(self) -> Dict[str, str]:
        """Get rate limit headers for HTTP response."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


@dataclass
class ClientQuota:
    """Quota tracking for a client."""

    client_id: str
    hourly_limit: int
    hourly_used: int = 0
    hour_start: float = field(default_factory=time.time)
    daily_limit: int = 0
    daily_used: int = 0
    day_start: float = field(default_factory=time.time)

    def check_quota(self, cost: int = 1) -> Tuple[bool, int]:
        """Check if quota allows request."""
        now = time.time()

        # Reset hourly if needed
        if now - self.hour_start >= 3600:
            self.hourly_used = 0
            self.hour_start = now

        # Reset daily if needed
        if now - self.day_start >= 86400:
            self.daily_used = 0
            self.day_start = now

        # Check limits
        if self.hourly_limit > 0 and self.hourly_used + cost > self.hourly_limit:
            return False, self.hourly_limit - self.hourly_used

        if self.daily_limit > 0 and self.daily_used + cost > self.daily_limit:
            return False, self.daily_limit - self.daily_used

        return True, self.hourly_limit - self.hourly_used - cost

    def consume(self, cost: int = 1) -> None:
        """Consume quota."""
        self.hourly_used += cost
        self.daily_used += cost


# ============================================================================
# Rate Limiter Backends
# ============================================================================


class RateLimiterBackend(ABC):
    """Abstract base for rate limiter backends."""

    @abstractmethod
    def allow(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed."""
        pass

    @abstractmethod
    def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        pass


class TokenBucket(RateLimiterBackend):
    """
    Token bucket rate limiter.

    Allows bursts up to bucket size, then refills at a constant rate.
    Good for allowing burst traffic while maintaining average rate.
    """

    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: int,  # bucket size
    ):
        self.rate = rate
        self.capacity = capacity

        # State per key
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: int = 1) -> RateLimitResult:
        now = time.time()

        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": float(self.capacity),
                    "last_update": now,
                }

            bucket = self._buckets[key]

            # Refill tokens
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(
                self.capacity,
                bucket["tokens"] + elapsed * self.rate
            )
            bucket["last_update"] = now

            if bucket["tokens"] >= cost:
                bucket["tokens"] -= cost
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket["tokens"]),
                    reset_at=now + (self.capacity - bucket["tokens"]) / self.rate,
                    limit=self.capacity,
                    client_id=key,
                )
            else:
                wait_time = (cost - bucket["tokens"]) / self.rate
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=now + wait_time,
                    retry_after=wait_time,
                    limit=self.capacity,
                    client_id=key,
                )

    def reset(self, key: str) -> None:
        with self._lock:
            if key in self._buckets:
                self._buckets[key]["tokens"] = float(self.capacity)
                self._buckets[key]["last_update"] = time.time()


class SlidingWindow(RateLimiterBackend):
    """
    Sliding window rate limiter.

    Tracks requests in a sliding time window.
    More accurate than fixed window but uses more memory.
    """

    def __init__(
        self,
        limit: int,  # requests per window
        window_seconds: int,  # window size
    ):
        self.limit = limit
        self.window_seconds = window_seconds

        # Request timestamps per key
        self._windows: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: int = 1) -> RateLimitResult:
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            if key not in self._windows:
                self._windows[key] = []

            # Remove old entries
            self._windows[key] = [
                ts for ts in self._windows[key]
                if ts > window_start
            ]

            current_count = len(self._windows[key])

            if current_count + cost <= self.limit:
                # Add timestamps for this request
                for _ in range(cost):
                    self._windows[key].append(now)

                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - current_count - cost,
                    reset_at=now + self.window_seconds,
                    limit=self.limit,
                    client_id=key,
                )
            else:
                # Calculate when oldest request expires
                oldest = min(self._windows[key]) if self._windows[key] else now
                retry_after = oldest + self.window_seconds - now

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=oldest + self.window_seconds,
                    retry_after=max(0, retry_after),
                    limit=self.limit,
                    client_id=key,
                )

    def reset(self, key: str) -> None:
        with self._lock:
            self._windows[key] = []


class FixedWindow(RateLimiterBackend):
    """
    Fixed window rate limiter.

    Simple counter that resets at fixed intervals.
    May allow burst at window boundaries.
    """

    def __init__(
        self,
        limit: int,  # requests per window
        window_seconds: int,  # window size
    ):
        self.limit = limit
        self.window_seconds = window_seconds

        # Counters per key
        self._counters: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: int = 1) -> RateLimitResult:
        now = time.time()
        window_id = int(now // self.window_seconds)

        with self._lock:
            if key not in self._counters:
                self._counters[key] = {
                    "window_id": window_id,
                    "count": 0,
                }

            counter = self._counters[key]

            # Reset if new window
            if counter["window_id"] != window_id:
                counter["window_id"] = window_id
                counter["count"] = 0

            if counter["count"] + cost <= self.limit:
                counter["count"] += cost

                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - counter["count"],
                    reset_at=(window_id + 1) * self.window_seconds,
                    limit=self.limit,
                    client_id=key,
                )
            else:
                reset_at = (window_id + 1) * self.window_seconds

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=reset_at - now,
                    limit=self.limit,
                    client_id=key,
                )

    def reset(self, key: str) -> None:
        with self._lock:
            if key in self._counters:
                self._counters[key]["count"] = 0


class LeakyBucket(RateLimiterBackend):
    """
    Leaky bucket rate limiter.

    Processes requests at a fixed rate, queuing excess.
    Good for smoothing traffic.
    """

    def __init__(
        self,
        rate: float,  # requests per second
        capacity: int,  # queue size
    ):
        self.rate = rate
        self.capacity = capacity

        # State per key
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: int = 1) -> RateLimitResult:
        now = time.time()

        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = {
                    "water": 0.0,
                    "last_update": now,
                }

            bucket = self._buckets[key]

            # Leak water
            elapsed = now - bucket["last_update"]
            bucket["water"] = max(0, bucket["water"] - elapsed * self.rate)
            bucket["last_update"] = now

            if bucket["water"] + cost <= self.capacity:
                bucket["water"] += cost

                return RateLimitResult(
                    allowed=True,
                    remaining=int(self.capacity - bucket["water"]),
                    reset_at=now + bucket["water"] / self.rate,
                    limit=self.capacity,
                    client_id=key,
                )
            else:
                wait_time = (bucket["water"] + cost - self.capacity) / self.rate

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=now + wait_time,
                    retry_after=wait_time,
                    limit=self.capacity,
                    client_id=key,
                )

    def reset(self, key: str) -> None:
        with self._lock:
            if key in self._buckets:
                self._buckets[key]["water"] = 0
                self._buckets[key]["last_update"] = time.time()


# ============================================================================
# Rate Limiter
# ============================================================================


class RateLimiter:
    """
    API rate limiter with multiple algorithms and quota support.

    Features:
    - Multiple algorithms (token bucket, sliding window, etc.)
    - Per-client limits
    - Quota tracking
    - Bus event emission

    PBTSO Phase: PROTECT

    Example:
        limiter = RateLimiter()

        # Check rate limit
        result = limiter.check("client-123")
        if not result.allowed:
            raise RateLimitExceeded(retry_after=result.retry_after)

        # With decorator
        @limiter.limit(key_fn=lambda req: req.client_id)
        async def handle_request(req):
            ...
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            config: Rate limiter configuration
            bus: AgentBus for event emission
        """
        self.config = config or RateLimitConfig()
        self.bus = bus or AgentBus()

        # Create backend
        self._backend = self._create_backend()

        # Quota tracking
        self._quotas: Dict[str, ClientQuota] = {}
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rejected_requests": 0,
        }

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """
        Check if request is allowed.

        Args:
            key: Client or resource identifier
            cost: Request cost (default 1)

        Returns:
            RateLimitResult with allowed status
        """
        self._stats["total_requests"] += 1

        # Check quota first
        if self.config.enable_quota:
            quota_ok, remaining = self._check_quota(key, cost)
            if not quota_ok:
                self._stats["rejected_requests"] += 1
                self._emit_exceeded(key, "quota", remaining)
                return RateLimitResult(
                    allowed=False,
                    remaining=remaining,
                    reset_at=time.time() + 3600,  # Reset at next hour
                    retry_after=3600,
                    limit=self.config.quota_per_hour,
                    client_id=key,
                )

        # Check rate limit
        result = self._backend.allow(key, cost)

        if result.allowed:
            self._stats["allowed_requests"] += 1

            # Consume quota
            if self.config.enable_quota:
                self._consume_quota(key, cost)
        else:
            self._stats["rejected_requests"] += 1
            self._emit_exceeded(key, "rate", result.remaining)

        return result

    def allow(self, key: str, cost: int = 1) -> bool:
        """
        Check if request is allowed (simple boolean).

        Args:
            key: Client identifier
            cost: Request cost

        Returns:
            True if allowed
        """
        return self.check(key, cost).allowed

    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        self._backend.reset(key)

        with self._lock:
            if key in self._quotas:
                self._quotas[key].hourly_used = 0
                self._quotas[key].hour_start = time.time()

        self._emit_reset(key)

    def limit(
        self,
        key_fn: Optional[Callable[..., str]] = None,
        cost_fn: Optional[Callable[..., int]] = None,
        on_exceeded: Optional[Callable[[RateLimitResult], None]] = None,
    ) -> Callable:
        """
        Decorator to rate limit a function.

        Args:
            key_fn: Function to extract key from arguments
            cost_fn: Function to calculate cost from arguments
            on_exceeded: Callback when rate limit exceeded

        Example:
            @limiter.limit(key_fn=lambda req: req.client_id)
            def handle(req):
                ...
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Extract key
                if key_fn:
                    key = key_fn(*args, **kwargs)
                else:
                    key = "default"

                # Calculate cost
                if cost_fn:
                    cost = cost_fn(*args, **kwargs)
                else:
                    cost = 1

                # Check limit
                result = self.check(key, cost)

                if not result.allowed:
                    if on_exceeded:
                        on_exceeded(result)
                    raise RateLimitExceeded(result)

                return func(*args, **kwargs)

            return wrapper
        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        total = self._stats["total_requests"]
        return {
            **self._stats,
            "rejection_rate": (
                self._stats["rejected_requests"] / total
                if total > 0 else 0.0
            ),
            "algorithm": self.config.algorithm.value,
        }

    def get_quota(self, key: str) -> Optional[ClientQuota]:
        """Get quota for a client."""
        return self._quotas.get(key)

    def set_quota(
        self,
        key: str,
        hourly_limit: int,
        daily_limit: int = 0,
    ) -> None:
        """Set quota for a client."""
        with self._lock:
            self._quotas[key] = ClientQuota(
                client_id=key,
                hourly_limit=hourly_limit,
                daily_limit=daily_limit,
            )

    def _create_backend(self) -> RateLimiterBackend:
        """Create rate limiter backend."""
        if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return TokenBucket(
                rate=self.config.requests_per_second,
                capacity=self.config.burst_size,
            )
        elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return SlidingWindow(
                limit=int(self.config.requests_per_minute),
                window_seconds=self.config.window_size_seconds,
            )
        elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return FixedWindow(
                limit=int(self.config.requests_per_minute),
                window_seconds=self.config.window_size_seconds,
            )
        elif self.config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return LeakyBucket(
                rate=self.config.requests_per_second,
                capacity=self.config.burst_size,
            )
        else:
            return TokenBucket(
                rate=self.config.requests_per_second,
                capacity=self.config.burst_size,
            )

    def _check_quota(self, key: str, cost: int) -> Tuple[bool, int]:
        """Check client quota."""
        with self._lock:
            if key not in self._quotas:
                self._quotas[key] = ClientQuota(
                    client_id=key,
                    hourly_limit=self.config.quota_per_hour,
                )

            return self._quotas[key].check_quota(cost)

    def _consume_quota(self, key: str, cost: int) -> None:
        """Consume client quota."""
        with self._lock:
            if key in self._quotas:
                self._quotas[key].consume(cost)

    def _emit_exceeded(self, key: str, limit_type: str, remaining: int) -> None:
        """Emit rate limit exceeded event."""
        if not self.config.emit_to_bus:
            return

        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": "a2a.research.rate_limit.exceeded",
            "kind": "rate_limit",
            "level": "warning",
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "client_id": key,
                "limit_type": limit_type,
                "remaining": remaining,
            },
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _emit_reset(self, key: str) -> None:
        """Emit rate limit reset event."""
        if not self.config.emit_to_bus:
            return

        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": "a2a.research.rate_limit.reset",
            "kind": "rate_limit",
            "level": "info",
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {"client_id": key},
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, result: RateLimitResult):
        self.result = result
        super().__init__(f"Rate limit exceeded. Retry after {result.retry_after}s")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Rate Limiter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Rate Limiter (Step 38)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Check command
    check_parser = subparsers.add_parser("check", help="Check rate limit")
    check_parser.add_argument("key", help="Client key")
    check_parser.add_argument("--cost", type=int, default=1, help="Request cost")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset rate limit")
    reset_parser.add_argument("key", help="Client key")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run rate limiter demo")
    demo_parser.add_argument("--algorithm", choices=["token_bucket", "sliding_window", "fixed_window"])

    args = parser.parse_args()

    config = RateLimitConfig()
    if hasattr(args, "algorithm") and args.algorithm:
        config.algorithm = RateLimitAlgorithm(args.algorithm)

    limiter = RateLimiter(config)

    if args.command == "check":
        result = limiter.check(args.key, args.cost)
        print(f"Allowed: {result.allowed}")
        print(f"Remaining: {result.remaining}")
        print(f"Reset At: {datetime.fromtimestamp(result.reset_at)}")
        if result.retry_after:
            print(f"Retry After: {result.retry_after:.1f}s")

        return 0 if result.allowed else 1

    elif args.command == "stats":
        stats = limiter.get_stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Rate Limiter Statistics:")
            print(f"  Algorithm: {stats['algorithm']}")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Allowed: {stats['allowed_requests']}")
            print(f"  Rejected: {stats['rejected_requests']}")
            print(f"  Rejection Rate: {stats['rejection_rate']:.1%}")

    elif args.command == "reset":
        limiter.reset(args.key)
        print(f"Rate limit reset for '{args.key}'")

    elif args.command == "demo":
        print(f"Running rate limiter demo ({config.algorithm.value})...\n")
        print(f"Config: {config.requests_per_second} req/s, burst={config.burst_size}\n")

        # Simulate requests
        for i in range(30):
            result = limiter.check("demo-client")
            status = "OK" if result.allowed else "BLOCKED"
            print(f"Request {i+1}: {status} (remaining: {result.remaining})")

            if not result.allowed:
                print(f"  -> Retry after {result.retry_after:.2f}s")
                time.sleep(result.retry_after)

            time.sleep(0.05)  # 50ms between requests

        print("\nDemo complete.")
        stats = limiter.get_stats()
        print(f"Total: {stats['total_requests']}, Allowed: {stats['allowed_requests']}, Rejected: {stats['rejected_requests']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
