#!/usr/bin/env python3
"""
limiter.py - Deploy Rate Limiter (Step 238)

PBTSO Phase: SKILL, SEQUESTER
A2A Integration: API rate limiting for deployments via deploy.ratelimit.*

Provides:
- RateLimitAlgorithm: Rate limiting algorithms
- RateLimitScope: Rate limit scope levels
- RateLimitConfig: Rate limit configuration
- RateLimitResult: Rate limit check result
- RateLimitBucket: Token bucket implementation
- DeployRateLimiter: Main rate limiter

Bus Topics:
- deploy.ratelimit.allowed
- deploy.ratelimit.denied
- deploy.ratelimit.throttled

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ==============================================================================
# Bus Emission Helper with File Locking (DKIN v30)
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "rate-limiter"
) -> str:
    """Emit an event to the Pluribus bus with fcntl.flock() file locking."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(Enum):
    """Rate limit scope levels."""
    GLOBAL = "global"
    USER = "user"
    SERVICE = "service"
    ENDPOINT = "endpoint"
    IP = "ip"


class RateLimitAction(Enum):
    """Actions when rate limit is exceeded."""
    REJECT = "reject"
    QUEUE = "queue"
    THROTTLE = "throttle"


@dataclass
class RateLimitConfig:
    """
    Rate limit configuration.

    Attributes:
        config_id: Configuration identifier
        name: Configuration name
        algorithm: Rate limiting algorithm
        scope: Rate limit scope
        requests: Maximum requests
        period_s: Time period in seconds
        burst: Burst capacity (for token bucket)
        action: Action when exceeded
        enabled: Whether limit is enabled
        endpoints: Specific endpoints (empty = all)
        users: Specific users (empty = all)
        services: Specific services (empty = all)
    """
    config_id: str
    name: str
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    scope: RateLimitScope = RateLimitScope.GLOBAL
    requests: int = 100
    period_s: int = 60
    burst: int = 0
    action: RateLimitAction = RateLimitAction.REJECT
    enabled: bool = True
    endpoints: List[str] = field(default_factory=list)
    users: List[str] = field(default_factory=list)
    services: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.burst == 0:
            self.burst = self.requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "name": self.name,
            "algorithm": self.algorithm.value,
            "scope": self.scope.value,
            "requests": self.requests,
            "period_s": self.period_s,
            "burst": self.burst,
            "action": self.action.value,
            "enabled": self.enabled,
            "endpoints": self.endpoints,
            "users": self.users,
            "services": self.services,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RateLimitConfig":
        data = dict(data)
        if "algorithm" in data:
            data["algorithm"] = RateLimitAlgorithm(data["algorithm"])
        if "scope" in data:
            data["scope"] = RateLimitScope(data["scope"])
        if "action" in data:
            data["action"] = RateLimitAction(data["action"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RateLimitResult:
    """
    Rate limit check result.

    Attributes:
        allowed: Whether request is allowed
        remaining: Remaining requests
        limit: Request limit
        reset_at: When limit resets
        retry_after_s: Seconds to wait before retry
        config_id: Applied configuration ID
        reason: Reason if denied
    """
    allowed: bool
    remaining: int
    limit: int
    reset_at: float
    retry_after_s: int = 0
    config_id: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RateLimitBucket:
    """
    Token bucket for rate limiting.

    Attributes:
        key: Bucket key
        tokens: Current token count
        last_update: Last token update time
        window_start: Current window start time
        request_count: Requests in current window
        history: Request timestamps (for sliding window)
    """
    key: str
    tokens: float = 0.0
    last_update: float = field(default_factory=time.time)
    window_start: float = field(default_factory=time.time)
    request_count: int = 0
    history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "tokens": self.tokens,
            "last_update": self.last_update,
            "window_start": self.window_start,
            "request_count": self.request_count,
        }


@dataclass
class RateLimitStats:
    """Rate limit statistics."""
    total_requests: int = 0
    allowed_requests: int = 0
    denied_requests: int = 0
    throttled_requests: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Deploy Rate Limiter (Step 238)
# ==============================================================================

class DeployRateLimiter:
    """
    Deploy Rate Limiter - API rate limiting for deployments.

    PBTSO Phase: SKILL, SEQUESTER

    Responsibilities:
    - Enforce rate limits on deployment operations
    - Support multiple rate limiting algorithms
    - Track rate limit statistics
    - Handle burst traffic
    - Support per-user, per-service limits

    Example:
        >>> limiter = DeployRateLimiter()
        >>> config = RateLimitConfig(
        ...     config_id="deploy-limit",
        ...     name="Deployment Rate Limit",
        ...     requests=10,
        ...     period_s=60,
        ... )
        >>> limiter.add_config(config)
        >>> result = limiter.check_rate_limit("user-123", "deploy")
        >>> if result.allowed:
        ...     # proceed with deployment
        ...     pass
    """

    BUS_TOPICS = {
        "allowed": "deploy.ratelimit.allowed",
        "denied": "deploy.ratelimit.denied",
        "throttled": "deploy.ratelimit.throttled",
    }

    # A2A heartbeat (CITIZEN v2)
    HEARTBEAT_INTERVAL_S = 300
    HEARTBEAT_TIMEOUT_S = 900

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "rate-limiter",
        cleanup_interval_s: int = 300,
    ):
        """
        Initialize the rate limiter.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            cleanup_interval_s: Interval for cleaning up old buckets
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "ratelimit"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.cleanup_interval_s = cleanup_interval_s

        # Rate limit configurations
        self._configs: Dict[str, RateLimitConfig] = {}

        # Token buckets by key
        self._buckets: Dict[str, RateLimitBucket] = {}

        # Statistics
        self._stats: Dict[str, RateLimitStats] = defaultdict(RateLimitStats)
        self._global_stats = RateLimitStats()

        # Queued requests (for QUEUE action)
        self._queue: List[Dict[str, Any]] = []

        self._last_cleanup = time.time()

        self._load_state()

    def add_config(self, config: RateLimitConfig) -> None:
        """Add a rate limit configuration."""
        self._configs[config.config_id] = config
        self._save_state()

    def remove_config(self, config_id: str) -> bool:
        """Remove a rate limit configuration."""
        if config_id in self._configs:
            del self._configs[config_id]
            self._save_state()
            return True
        return False

    def get_config(self, config_id: str) -> Optional[RateLimitConfig]:
        """Get a rate limit configuration."""
        return self._configs.get(config_id)

    def check_rate_limit(
        self,
        key: str,
        endpoint: str = "",
        user: str = "",
        service: str = "",
        cost: int = 1,
    ) -> RateLimitResult:
        """
        Check if a request is allowed under rate limits.

        Args:
            key: Rate limit key (e.g., user ID, IP)
            endpoint: Request endpoint
            user: User identifier
            service: Service name
            cost: Request cost (default: 1)

        Returns:
            RateLimitResult
        """
        self._maybe_cleanup()

        # Find applicable configs
        applicable_configs = self._find_applicable_configs(key, endpoint, user, service)

        if not applicable_configs:
            return RateLimitResult(
                allowed=True,
                remaining=float("inf"),
                limit=0,
                reset_at=0,
            )

        # Check each applicable config
        most_restrictive: Optional[RateLimitResult] = None

        for config in applicable_configs:
            bucket_key = self._build_bucket_key(config, key, endpoint, user, service)
            result = self._check_bucket(config, bucket_key, cost)

            if not result.allowed:
                # Immediately return denied result
                self._record_denied(config, key)
                return result

            if most_restrictive is None or result.remaining < most_restrictive.remaining:
                most_restrictive = result

        if most_restrictive:
            self._record_allowed(most_restrictive.config_id, key)
            return most_restrictive

        return RateLimitResult(
            allowed=True,
            remaining=0,
            limit=0,
            reset_at=0,
        )

    async def check_rate_limit_async(
        self,
        key: str,
        endpoint: str = "",
        user: str = "",
        service: str = "",
        cost: int = 1,
        wait: bool = False,
        max_wait_s: int = 30,
    ) -> RateLimitResult:
        """
        Async version with optional wait for throttling.

        Args:
            key: Rate limit key
            endpoint: Request endpoint
            user: User identifier
            service: Service name
            cost: Request cost
            wait: Whether to wait if rate limited
            max_wait_s: Maximum wait time

        Returns:
            RateLimitResult
        """
        result = self.check_rate_limit(key, endpoint, user, service, cost)

        if not result.allowed and wait and result.retry_after_s <= max_wait_s:
            # Wait and retry
            await asyncio.sleep(result.retry_after_s)
            result = self.check_rate_limit(key, endpoint, user, service, cost)

            if result.allowed:
                self._record_throttled(result.config_id, key)

        return result

    def _find_applicable_configs(
        self,
        key: str,
        endpoint: str,
        user: str,
        service: str,
    ) -> List[RateLimitConfig]:
        """Find all applicable rate limit configurations."""
        applicable = []

        for config in self._configs.values():
            if not config.enabled:
                continue

            # Check endpoint filter
            if config.endpoints and endpoint and endpoint not in config.endpoints:
                continue

            # Check user filter
            if config.users and user and user not in config.users:
                continue

            # Check service filter
            if config.services and service and service not in config.services:
                continue

            applicable.append(config)

        return applicable

    def _build_bucket_key(
        self,
        config: RateLimitConfig,
        key: str,
        endpoint: str,
        user: str,
        service: str,
    ) -> str:
        """Build bucket key based on scope."""
        parts = [config.config_id]

        if config.scope == RateLimitScope.GLOBAL:
            pass
        elif config.scope == RateLimitScope.USER:
            parts.append(f"user:{user or key}")
        elif config.scope == RateLimitScope.SERVICE:
            parts.append(f"service:{service or key}")
        elif config.scope == RateLimitScope.ENDPOINT:
            parts.append(f"endpoint:{endpoint or key}")
        elif config.scope == RateLimitScope.IP:
            parts.append(f"ip:{key}")

        return ":".join(parts)

    def _check_bucket(
        self,
        config: RateLimitConfig,
        bucket_key: str,
        cost: int,
    ) -> RateLimitResult:
        """Check and update a rate limit bucket."""
        bucket = self._get_or_create_bucket(bucket_key, config)

        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return self._check_token_bucket(config, bucket, cost)
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return self._check_sliding_window(config, bucket, cost)
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return self._check_fixed_window(config, bucket, cost)
        elif config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return self._check_leaky_bucket(config, bucket, cost)
        else:
            return RateLimitResult(
                allowed=True,
                remaining=config.requests,
                limit=config.requests,
                reset_at=time.time() + config.period_s,
            )

    def _get_or_create_bucket(
        self,
        bucket_key: str,
        config: RateLimitConfig,
    ) -> RateLimitBucket:
        """Get or create a rate limit bucket."""
        if bucket_key not in self._buckets:
            self._buckets[bucket_key] = RateLimitBucket(
                key=bucket_key,
                tokens=float(config.burst),
                last_update=time.time(),
                window_start=time.time(),
            )
        return self._buckets[bucket_key]

    def _check_token_bucket(
        self,
        config: RateLimitConfig,
        bucket: RateLimitBucket,
        cost: int,
    ) -> RateLimitResult:
        """Token bucket algorithm."""
        now = time.time()

        # Refill tokens
        time_passed = now - bucket.last_update
        refill_rate = config.requests / config.period_s
        bucket.tokens = min(config.burst, bucket.tokens + time_passed * refill_rate)
        bucket.last_update = now

        # Check if we have enough tokens
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return RateLimitResult(
                allowed=True,
                remaining=int(bucket.tokens),
                limit=config.requests,
                reset_at=now + config.period_s,
                config_id=config.config_id,
            )
        else:
            # Calculate retry after
            tokens_needed = cost - bucket.tokens
            retry_after = int(tokens_needed / refill_rate) + 1

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=config.requests,
                reset_at=now + retry_after,
                retry_after_s=retry_after,
                config_id=config.config_id,
                reason="Rate limit exceeded",
            )

    def _check_sliding_window(
        self,
        config: RateLimitConfig,
        bucket: RateLimitBucket,
        cost: int,
    ) -> RateLimitResult:
        """Sliding window algorithm."""
        now = time.time()
        window_start = now - config.period_s

        # Remove old entries
        bucket.history = [ts for ts in bucket.history if ts > window_start]

        # Check limit
        if len(bucket.history) + cost <= config.requests:
            # Add new requests
            for _ in range(cost):
                bucket.history.append(now)

            return RateLimitResult(
                allowed=True,
                remaining=config.requests - len(bucket.history),
                limit=config.requests,
                reset_at=bucket.history[0] + config.period_s if bucket.history else now + config.period_s,
                config_id=config.config_id,
            )
        else:
            # Calculate when oldest entry expires
            retry_after = int(bucket.history[0] + config.period_s - now) + 1 if bucket.history else 1

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=config.requests,
                reset_at=bucket.history[0] + config.period_s if bucket.history else now + config.period_s,
                retry_after_s=max(1, retry_after),
                config_id=config.config_id,
                reason="Rate limit exceeded",
            )

    def _check_fixed_window(
        self,
        config: RateLimitConfig,
        bucket: RateLimitBucket,
        cost: int,
    ) -> RateLimitResult:
        """Fixed window algorithm."""
        now = time.time()

        # Check if we're in a new window
        if now - bucket.window_start >= config.period_s:
            bucket.window_start = now
            bucket.request_count = 0

        # Check limit
        if bucket.request_count + cost <= config.requests:
            bucket.request_count += cost

            return RateLimitResult(
                allowed=True,
                remaining=config.requests - bucket.request_count,
                limit=config.requests,
                reset_at=bucket.window_start + config.period_s,
                config_id=config.config_id,
            )
        else:
            retry_after = int(bucket.window_start + config.period_s - now) + 1

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=config.requests,
                reset_at=bucket.window_start + config.period_s,
                retry_after_s=max(1, retry_after),
                config_id=config.config_id,
                reason="Rate limit exceeded",
            )

    def _check_leaky_bucket(
        self,
        config: RateLimitConfig,
        bucket: RateLimitBucket,
        cost: int,
    ) -> RateLimitResult:
        """Leaky bucket algorithm (queue-based)."""
        now = time.time()

        # Calculate leak rate
        leak_rate = config.requests / config.period_s
        time_passed = now - bucket.last_update

        # Leak tokens
        leaked = time_passed * leak_rate
        bucket.tokens = max(0, bucket.tokens - leaked)
        bucket.last_update = now

        # Check if we can add to bucket
        if bucket.tokens + cost <= config.burst:
            bucket.tokens += cost

            return RateLimitResult(
                allowed=True,
                remaining=int(config.burst - bucket.tokens),
                limit=config.requests,
                reset_at=now + (bucket.tokens / leak_rate),
                config_id=config.config_id,
            )
        else:
            # Queue is full
            wait_time = (bucket.tokens + cost - config.burst) / leak_rate

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=config.requests,
                reset_at=now + wait_time,
                retry_after_s=int(wait_time) + 1,
                config_id=config.config_id,
                reason="Rate limit exceeded (queue full)",
            )

    def _record_allowed(self, config_id: str, key: str) -> None:
        """Record an allowed request."""
        self._global_stats.total_requests += 1
        self._global_stats.allowed_requests += 1
        self._stats[config_id].total_requests += 1
        self._stats[config_id].allowed_requests += 1

        _emit_bus_event(
            self.BUS_TOPICS["allowed"],
            {
                "config_id": config_id,
                "key": key,
            },
            kind="metric",
            actor=self.actor_id,
        )

    def _record_denied(self, config: RateLimitConfig, key: str) -> None:
        """Record a denied request."""
        self._global_stats.total_requests += 1
        self._global_stats.denied_requests += 1
        self._stats[config.config_id].total_requests += 1
        self._stats[config.config_id].denied_requests += 1

        _emit_bus_event(
            self.BUS_TOPICS["denied"],
            {
                "config_id": config.config_id,
                "config_name": config.name,
                "key": key,
                "action": config.action.value,
            },
            level="warn",
            actor=self.actor_id,
        )

    def _record_throttled(self, config_id: str, key: str) -> None:
        """Record a throttled request."""
        self._global_stats.throttled_requests += 1
        self._stats[config_id].throttled_requests += 1

        _emit_bus_event(
            self.BUS_TOPICS["throttled"],
            {
                "config_id": config_id,
                "key": key,
            },
            kind="metric",
            actor=self.actor_id,
        )

    def _maybe_cleanup(self) -> None:
        """Clean up old buckets periodically."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval_s:
            return

        self._last_cleanup = now
        cutoff = now - 3600  # Remove buckets older than 1 hour

        keys_to_remove = []
        for key, bucket in self._buckets.items():
            if bucket.last_update < cutoff:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._buckets[key]

    def get_stats(self, config_id: Optional[str] = None) -> RateLimitStats:
        """Get rate limit statistics."""
        if config_id:
            return self._stats.get(config_id, RateLimitStats())
        return self._global_stats

    def get_bucket_info(self, bucket_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific bucket."""
        bucket = self._buckets.get(bucket_key)
        if bucket:
            return bucket.to_dict()
        return None

    def reset_bucket(self, bucket_key: str) -> bool:
        """Reset a specific bucket."""
        if bucket_key in self._buckets:
            del self._buckets[bucket_key]
            return True
        return False

    def reset_stats(self, config_id: Optional[str] = None) -> None:
        """Reset statistics."""
        if config_id:
            self._stats[config_id] = RateLimitStats()
        else:
            self._stats.clear()
            self._global_stats = RateLimitStats()

    def list_configs(self) -> List[RateLimitConfig]:
        """List all rate limit configurations."""
        return list(self._configs.values())

    def list_buckets(self) -> List[str]:
        """List all bucket keys."""
        return list(self._buckets.keys())

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "configs": {k: v.to_dict() for k, v in self._configs.items()},
        }
        state_file = self.state_dir / "ratelimit_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "ratelimit_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for config_id, data in state.get("configs", {}).items():
                self._configs[config_id] = RateLimitConfig.from_dict(data)

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for rate limiter."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Rate Limiter (Step 238)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check command
    check_parser = subparsers.add_parser("check", help="Check rate limit")
    check_parser.add_argument("key", help="Rate limit key")
    check_parser.add_argument("--endpoint", "-e", default="", help="Endpoint")
    check_parser.add_argument("--user", "-u", default="", help="User")
    check_parser.add_argument("--service", "-s", default="", help="Service")
    check_parser.add_argument("--cost", "-c", type=int, default=1, help="Request cost")
    check_parser.add_argument("--json", action="store_true", help="JSON output")

    # add-config command
    add_parser = subparsers.add_parser("add-config", help="Add rate limit config")
    add_parser.add_argument("name", help="Config name")
    add_parser.add_argument("--requests", "-r", type=int, default=100, help="Max requests")
    add_parser.add_argument("--period", "-p", type=int, default=60, help="Period in seconds")
    add_parser.add_argument("--burst", "-b", type=int, default=0, help="Burst capacity")
    add_parser.add_argument("--algorithm", "-a", default="token_bucket",
                            choices=["token_bucket", "sliding_window", "fixed_window", "leaky_bucket"])
    add_parser.add_argument("--scope", default="global",
                            choices=["global", "user", "service", "endpoint", "ip"])

    # list-configs command
    list_parser = subparsers.add_parser("list-configs", help="List rate limit configs")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # remove-config command
    remove_parser = subparsers.add_parser("remove-config", help="Remove rate limit config")
    remove_parser.add_argument("config_id", help="Config ID")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Get rate limit stats")
    stats_parser.add_argument("--config", "-c", help="Config ID")
    stats_parser.add_argument("--json", action="store_true", help="JSON output")

    # buckets command
    buckets_parser = subparsers.add_parser("buckets", help="List buckets")
    buckets_parser.add_argument("--json", action="store_true", help="JSON output")

    # reset command
    reset_parser = subparsers.add_parser("reset", help="Reset bucket or stats")
    reset_parser.add_argument("--bucket", "-b", help="Bucket key to reset")
    reset_parser.add_argument("--stats", "-s", action="store_true", help="Reset stats")

    args = parser.parse_args()
    limiter = DeployRateLimiter()

    if args.command == "check":
        result = limiter.check_rate_limit(
            args.key,
            endpoint=args.endpoint,
            user=args.user,
            service=args.service,
            cost=args.cost,
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "ALLOWED" if result.allowed else "DENIED"
            print(f"[{status}] Rate limit check")
            print(f"  Remaining: {result.remaining}/{result.limit}")
            if not result.allowed:
                print(f"  Retry after: {result.retry_after_s}s")
                print(f"  Reason: {result.reason}")

        return 0 if result.allowed else 1

    elif args.command == "add-config":
        config = RateLimitConfig(
            config_id=f"limit-{uuid.uuid4().hex[:8]}",
            name=args.name,
            requests=args.requests,
            period_s=args.period,
            burst=args.burst or args.requests,
            algorithm=RateLimitAlgorithm(args.algorithm),
            scope=RateLimitScope(args.scope),
        )
        limiter.add_config(config)
        print(f"Added config: {config.config_id}")
        print(f"  Name: {config.name}")
        print(f"  Limit: {config.requests} requests / {config.period_s}s")
        return 0

    elif args.command == "list-configs":
        configs = limiter.list_configs()

        if args.json:
            print(json.dumps([c.to_dict() for c in configs], indent=2))
        else:
            if not configs:
                print("No rate limit configs")
            else:
                for c in configs:
                    status = "enabled" if c.enabled else "disabled"
                    print(f"{c.config_id} ({c.name}): {c.requests}/{c.period_s}s [{c.algorithm.value}] [{status}]")

        return 0

    elif args.command == "remove-config":
        success = limiter.remove_config(args.config_id)
        if success:
            print(f"Removed config: {args.config_id}")
        else:
            print(f"Config not found: {args.config_id}")
            return 1
        return 0

    elif args.command == "stats":
        stats = limiter.get_stats(args.config)

        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print("Rate Limit Statistics:")
            print(f"  Total requests: {stats.total_requests}")
            print(f"  Allowed: {stats.allowed_requests}")
            print(f"  Denied: {stats.denied_requests}")
            print(f"  Throttled: {stats.throttled_requests}")

        return 0

    elif args.command == "buckets":
        buckets = limiter.list_buckets()

        if args.json:
            print(json.dumps(buckets, indent=2))
        else:
            print(f"Active buckets: {len(buckets)}")
            for b in buckets[:20]:
                print(f"  {b}")

        return 0

    elif args.command == "reset":
        if args.bucket:
            success = limiter.reset_bucket(args.bucket)
            if success:
                print(f"Reset bucket: {args.bucket}")
            else:
                print(f"Bucket not found: {args.bucket}")
                return 1
        if args.stats:
            limiter.reset_stats()
            print("Reset stats")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
