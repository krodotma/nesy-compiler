#!/usr/bin/env python3
"""
RGMA Bounded Domains: Unsupervised Improvement Regions

Implements bounded domain configuration for RGMA (Rhizome Godel Machine Alpha),
allowing self-improvement within safe, defined regions without human approval.

Reference: nucleus/specs/rhizome_godel_alpha.md (Section 7)

Usage:
    python3 rgma_domains.py check <path>
    python3 rgma_domains.py list
    python3 rgma_domains.py validate [--config <path>]
    python3 rgma_domains.py info <domain_id>

Constitution: All mutations in bounded domains MUST emit bus events for
audit trail. Rollbacks are automatic when CMP drops below domain floor.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

sys.dont_write_bytecode = True

# Type aliases
RiskLevel = Literal[0, 1, 2, 3]
MutationOperator = Literal[
    "prompt_edit",
    "tool_rebind",
    "subgraph_inline",
    "threshold_tune",
    "temperature_adjust",
    "retry_policy",
    "motif_toggle",
    "hgt_splice",
    "macro_rewrite",
]

# Default config path
DEFAULT_CONFIG_PATH = Path("/pluribus/.rgma-domains.yaml")

# PHI constant for CMP calculations
PHI = 1.618033988749895


def now_iso_utc() -> str:
    """Return current time in ISO 8601 UTC format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file, with fallback to pure-Python parser if PyYAML unavailable."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: simple YAML subset parser for our config format
        return _parse_yaml_fallback(path)


def _parse_yaml_fallback(path: Path) -> dict[str, Any]:
    """
    Minimal YAML parser for the domain config format.
    Handles: scalars, lists, nested dicts (by indentation).

    This is intentionally limited - for production, PyYAML should be installed.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, result)]
    current_list_key: Optional[str] = None

    for line in lines:
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Calculate indent level
        indent = len(line) - len(line.lstrip())

        # Handle list items
        if stripped.startswith("- "):
            content = stripped[2:].strip()
            # Pop to correct level
            while stack and stack[-1][0] >= indent:
                stack.pop()
            current_dict = stack[-1][1] if stack else result

            if ":" in content:
                # Dict item in list: "- id: value"
                # Find the parent list
                if current_list_key and current_list_key in current_dict:
                    item_dict: dict[str, Any] = {}
                    key, val = content.split(":", 1)
                    item_dict[key.strip()] = _parse_value(val.strip())
                    current_dict[current_list_key].append(item_dict)
                    stack.append((indent + 2, item_dict))
            else:
                # Simple list item: "- value"
                if current_list_key and current_list_key in current_dict:
                    current_dict[current_list_key].append(_parse_value(content))
        elif ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()

            # Pop to correct level
            while stack and stack[-1][0] >= indent:
                stack.pop()
            current_dict = stack[-1][1] if stack else result

            if not val:
                # Nested structure or list
                current_dict[key] = []
                current_list_key = key
                stack.append((indent, current_dict))
            elif val.startswith("[") and val.endswith("]"):
                # Inline list
                items = val[1:-1].split(",")
                current_dict[key] = [_parse_value(i.strip()) for i in items if i.strip()]
                current_list_key = None
            else:
                current_dict[key] = _parse_value(val)
                current_list_key = None

    return result


def _parse_value(val: str) -> Any:
    """Parse a YAML scalar value."""
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() in ("null", "~"):
        return None
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


@dataclass
class DomainMetrics:
    """Metrics configuration for a bounded domain."""
    track_mutations: bool = True
    alert_on_regression: bool = True
    regression_threshold: float = 0.1


@dataclass
class BoundedDomain:
    """
    A region where RGMA can self-improve without human approval.

    Implements Section 7.1 of rhizome_godel_alpha.md.
    """
    domain_id: str
    description: str = ""
    allowed_paths: list[str] = field(default_factory=list)
    forbidden_paths: list[str] = field(default_factory=list)
    allowed_operators: list[str] = field(default_factory=list)
    max_risk_level: int = 1
    cmp_floor: float = 0.5
    rollback_on_regression: bool = True
    metrics: DomainMetrics = field(default_factory=DomainMetrics)

    # Runtime state (not persisted)
    _regression_count: int = field(default=0, repr=False)
    _current_risk: int = field(default=-1, repr=False)

    def __post_init__(self) -> None:
        """Initialize runtime state."""
        if self._current_risk < 0:
            self._current_risk = self.max_risk_level

    def contains(self, path: str) -> bool:
        """
        Check if a file path is within this bounded domain.

        A path is contained if:
        1. It matches at least one allowed_path pattern
        2. It does NOT match any forbidden_path pattern

        Args:
            path: File path to check (relative or absolute)

        Returns:
            True if path is within this domain
        """
        # Normalize path - handle various path formats
        # Remove leading ./ but preserve leading . for hidden directories
        normalized = path
        if normalized.startswith("./"):
            normalized = normalized[2:]
        # Remove leading / for absolute paths
        normalized = normalized.lstrip("/")

        # Check forbidden paths first (exclusions take priority)
        for pattern in self.forbidden_paths:
            if fnmatch.fnmatch(normalized, pattern):
                return False
            # Also check with globstar expansion for **
            if "**" in pattern:
                # fnmatch doesn't handle ** well, so we expand it
                if self._matches_globstar(normalized, pattern):
                    return False

        # Check allowed paths
        for pattern in self.allowed_paths:
            if fnmatch.fnmatch(normalized, pattern):
                return True
            if "**" in pattern:
                if self._matches_globstar(normalized, pattern):
                    return True

        return False

    def _matches_globstar(self, path: str, pattern: str) -> bool:
        """
        Match a path against a pattern with ** (globstar).

        ** matches any number of directories (including zero).

        Examples:
            ".pluribus/motif_bank/**" matches ".pluribus/motif_bank/foo.json"
            ".pluribus/motif_bank/**/*.json" matches ".pluribus/motif_bank/sub/file.json"
            "nucleus/tools/**/*.config.json" matches "nucleus/tools/sub/tool.config.json"
        """
        import re

        # Convert glob pattern to regex
        # 1. Escape regex special chars except * and ?
        regex_pattern = ""
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c == "*":
                if i + 1 < len(pattern) and pattern[i + 1] == "*":
                    # ** - match any path segments (including none)
                    # Check if followed by /
                    if i + 2 < len(pattern) and pattern[i + 2] == "/":
                        regex_pattern += "(?:.*/?)?"
                        i += 3
                        continue
                    else:
                        # ** at end or followed by non-/
                        regex_pattern += ".*"
                        i += 2
                        continue
                else:
                    # Single * - match anything except /
                    regex_pattern += "[^/]*"
            elif c == "?":
                regex_pattern += "[^/]"
            elif c in ".^$+{}[]|()":
                regex_pattern += "\\" + c
            else:
                regex_pattern += c
            i += 1

        # Anchor the pattern
        regex_pattern = "^" + regex_pattern + "$"

        try:
            return bool(re.match(regex_pattern, path))
        except re.error:
            # Fallback to simple fnmatch if regex fails
            return fnmatch.fnmatch(path, pattern.replace("**", "*"))

    def is_operator_allowed(self, operator: str) -> bool:
        """Check if a mutation operator is allowed in this domain."""
        return operator in self.allowed_operators

    def get_effective_risk(self) -> int:
        """Get current risk level (may be escalated from base)."""
        return self._current_risk

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundedDomain":
        """Create a BoundedDomain from a dictionary (parsed YAML)."""
        metrics_data = data.get("metrics", {})
        metrics = DomainMetrics(
            track_mutations=metrics_data.get("track_mutations", True),
            alert_on_regression=metrics_data.get("alert_on_regression", True),
            regression_threshold=metrics_data.get("regression_threshold", 0.1),
        )

        return cls(
            domain_id=data.get("id", "unknown"),
            description=data.get("description", ""),
            allowed_paths=data.get("allowed_paths", []),
            forbidden_paths=data.get("forbidden_paths", []),
            allowed_operators=data.get("allowed_operators", []),
            max_risk_level=int(data.get("max_risk_level", 1)),
            cmp_floor=float(data.get("cmp_floor", 0.5)),
            rollback_on_regression=bool(data.get("rollback_on_regression", True)),
            metrics=metrics,
        )


@dataclass
class EscalationPolicy:
    """Risk escalation policy configuration."""
    regression_count_threshold: int = 3
    regression_window_seconds: int = 86400  # 24 hours
    max_escalation_level: int = 3
    cooldown_seconds: int = 604800  # 7 days

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EscalationPolicy":
        return cls(
            regression_count_threshold=int(data.get("regression_count_threshold", 3)),
            regression_window_seconds=int(data.get("regression_window_seconds", 86400)),
            max_escalation_level=int(data.get("max_escalation_level", 3)),
            cooldown_seconds=int(data.get("cooldown_seconds", 604800)),
        )


@dataclass
class GlobalSettings:
    """Global RGMA domain settings."""
    cmp_floor: float = 0.236  # 1/PHI^3 - extinction threshold
    default_risk: int = 3
    emit_events: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlobalSettings":
        return cls(
            cmp_floor=float(data.get("cmp_floor", 0.236)),
            default_risk=int(data.get("default_risk", 3)),
            emit_events=bool(data.get("emit_events", True)),
        )


@dataclass
class BusTopics:
    """Bus event topics for RGMA domain events."""
    rollback: str = "rgma.rollback"
    domain_violation: str = "rgma.domain.violation"
    risk_escalated: str = "rgma.risk.escalated"
    mutation_applied: str = "rgma.mutation.applied"
    domain_check: str = "rgma.domain.check"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BusTopics":
        return cls(
            rollback=data.get("rollback", "rgma.rollback"),
            domain_violation=data.get("domain_violation", "rgma.domain.violation"),
            risk_escalated=data.get("risk_escalated", "rgma.risk.escalated"),
            mutation_applied=data.get("mutation_applied", "rgma.mutation.applied"),
            domain_check=data.get("domain_check", "rgma.domain.check"),
        )


@dataclass
class AppliedMutation:
    """Record of an applied mutation for regression checking."""
    mutation_id: str
    domain_id: str
    operator: str
    target_path: str
    cmp_before: float
    cmp_after: Optional[float] = None
    commit_sha: Optional[str] = None
    applied_at: str = field(default_factory=now_iso_utc)
    rolled_back: bool = False
    rollback_reason: Optional[str] = None


class DomainManager:
    """
    Manages bounded domains for RGMA unsupervised improvement.

    Provides:
    - Domain lookup and path containment checks
    - Regression detection and auto-rollback
    - Risk escalation on repeated failures
    - Bus event emission for audit trail
    """

    def __init__(
        self,
        domains: list[BoundedDomain],
        global_settings: GlobalSettings,
        escalation_policy: EscalationPolicy,
        bus_topics: BusTopics,
    ) -> None:
        self.domains = {d.domain_id: d for d in domains}
        self.global_settings = global_settings
        self.escalation_policy = escalation_policy
        self.bus_topics = bus_topics

        # Runtime state
        self._regression_history: dict[str, list[float]] = {}  # domain_id -> timestamps
        self._bus_paths: Optional[Any] = None

    def _resolve_bus_paths(self) -> Any:
        """Lazily resolve bus paths for event emission."""
        if self._bus_paths is None:
            try:
                # Import agent_bus from the same directory
                tools_dir = Path(__file__).parent
                sys.path.insert(0, str(tools_dir))
                import agent_bus
                self._bus_paths = agent_bus.resolve_bus_paths(None)
            except Exception as e:
                # Create minimal fallback
                sys.stderr.write(f"WARN: Could not resolve bus paths: {e}\n")
                self._bus_paths = type("FallbackPaths", (), {
                    "events_path": "/pluribus/.pluribus/bus/events.ndjson",
                    "active_dir": "/pluribus/.pluribus/bus",
                })()
        return self._bus_paths

    def _emit_event(
        self,
        topic: str,
        kind: str,
        level: str,
        data: dict[str, Any],
        actor: str = "rgma-domains",
    ) -> Optional[str]:
        """Emit a bus event if event emission is enabled."""
        if not self.global_settings.emit_events:
            return None

        try:
            paths = self._resolve_bus_paths()
            tools_dir = Path(__file__).parent
            sys.path.insert(0, str(tools_dir))
            import agent_bus

            return agent_bus.emit_event(
                paths,
                topic=topic,
                kind=kind,
                level=level,
                actor=actor,
                data=data,
                trace_id=None,
                run_id=None,
                durable=True,
            )
        except Exception as e:
            sys.stderr.write(f"WARN: Failed to emit bus event: {e}\n")
            return None

    def contains(self, path: str) -> bool:
        """
        Check if a path is contained in any bounded domain.

        Args:
            path: File path to check

        Returns:
            True if path is in at least one domain
        """
        return any(domain.contains(path) for domain in self.domains.values())

    def get_domain(self, path: str) -> Optional[BoundedDomain]:
        """
        Get the bounded domain containing a path.

        If multiple domains match, returns the most specific one
        (smallest number of allowed_paths that match).

        Args:
            path: File path to check

        Returns:
            Matching BoundedDomain or None
        """
        matching_domains: list[BoundedDomain] = []

        for domain in self.domains.values():
            if domain.contains(path):
                matching_domains.append(domain)

        if not matching_domains:
            return None

        if len(matching_domains) == 1:
            return matching_domains[0]

        # Return most specific (fewest patterns that match)
        def specificity(d: BoundedDomain) -> int:
            return sum(1 for p in d.allowed_paths if fnmatch.fnmatch(path, p))

        return min(matching_domains, key=specificity)

    def get_domain_by_id(self, domain_id: str) -> Optional[BoundedDomain]:
        """Get a domain by its ID."""
        return self.domains.get(domain_id)

    def list_domains(self) -> list[BoundedDomain]:
        """List all configured domains."""
        return list(self.domains.values())

    def check_operator_allowed(
        self,
        path: str,
        operator: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if an operator is allowed for a path.

        Returns:
            Tuple of (allowed, reason). If not allowed, reason explains why.
        """
        domain = self.get_domain(path)

        if domain is None:
            return False, f"Path '{path}' is not in any bounded domain"

        if not domain.is_operator_allowed(operator):
            return False, (
                f"Operator '{operator}' not allowed in domain '{domain.domain_id}'. "
                f"Allowed: {domain.allowed_operators}"
            )

        return True, None

    def check_risk_level(
        self,
        path: str,
        proposed_risk: int,
    ) -> tuple[bool, int, Optional[str]]:
        """
        Check if a proposed risk level is within domain bounds.

        Returns:
            Tuple of (allowed, effective_risk, reason).
        """
        domain = self.get_domain(path)

        if domain is None:
            # Not in bounded domain - use global default
            return proposed_risk <= self.global_settings.default_risk, \
                   self.global_settings.default_risk, \
                   "Path not in bounded domain"

        effective_risk = domain.get_effective_risk()
        allowed = proposed_risk <= domain.max_risk_level

        reason = None
        if not allowed:
            reason = (
                f"Risk level {proposed_risk} exceeds domain max {domain.max_risk_level}"
            )

        return allowed, effective_risk, reason

    def check_regression(
        self,
        mutation: AppliedMutation,
        domain: Optional[BoundedDomain] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a mutation caused CMP regression and handle auto-rollback.

        Implements Section 7.3 of rhizome_godel_alpha.md:
        If CMP drops below domain floor after unsupervised mutation,
        auto-rollback and emit bus event.

        Args:
            mutation: The applied mutation to check
            domain: Optional domain (looked up if not provided)

        Returns:
            Tuple of (should_rollback, reason)
        """
        if domain is None:
            domain = self.get_domain(mutation.target_path)

        if domain is None:
            # Not in a bounded domain - no auto-rollback
            return False, None

        # Check if CMP dropped below floor
        if mutation.cmp_after is None:
            return False, None

        if mutation.cmp_after < domain.cmp_floor:
            reason = (
                f"CMP regression: {mutation.cmp_before:.3f} -> {mutation.cmp_after:.3f}, "
                f"below floor {domain.cmp_floor:.3f}"
            )

            if domain.rollback_on_regression:
                # Record regression for escalation tracking
                self._record_regression(domain.domain_id)

                # Emit rollback event
                self._emit_event(
                    topic=self.bus_topics.rollback,
                    kind="artifact",
                    level="warn",
                    data={
                        "mutation_id": mutation.mutation_id,
                        "domain_id": domain.domain_id,
                        "reason": "cmp_regression",
                        "cmp_before": mutation.cmp_before,
                        "cmp_after": mutation.cmp_after,
                        "floor": domain.cmp_floor,
                        "operator": mutation.operator,
                        "target_path": mutation.target_path,
                    },
                )

                # Check if escalation is needed
                self._check_escalation(domain)

                return True, reason
            else:
                # Domain doesn't auto-rollback (e.g., motif-library)
                return False, reason

        # Check relative regression threshold from metrics
        if domain.metrics.alert_on_regression and mutation.cmp_before > 0:
            relative_drop = (mutation.cmp_before - mutation.cmp_after) / mutation.cmp_before
            if relative_drop > domain.metrics.regression_threshold:
                reason = (
                    f"CMP relative drop: {relative_drop:.1%} exceeds threshold "
                    f"{domain.metrics.regression_threshold:.1%}"
                )
                # Alert but don't auto-rollback (above floor)
                self._emit_event(
                    topic="rgma.regression.alert",
                    kind="metric",
                    level="warn",
                    data={
                        "mutation_id": mutation.mutation_id,
                        "domain_id": domain.domain_id,
                        "cmp_before": mutation.cmp_before,
                        "cmp_after": mutation.cmp_after,
                        "relative_drop": relative_drop,
                        "threshold": domain.metrics.regression_threshold,
                    },
                )

        return False, None

    def _record_regression(self, domain_id: str) -> None:
        """Record a regression event for escalation tracking."""
        now = time.time()

        if domain_id not in self._regression_history:
            self._regression_history[domain_id] = []

        self._regression_history[domain_id].append(now)

        # Prune old entries outside the window
        cutoff = now - self.escalation_policy.regression_window_seconds
        self._regression_history[domain_id] = [
            ts for ts in self._regression_history[domain_id] if ts > cutoff
        ]

    def _check_escalation(self, domain: BoundedDomain) -> None:
        """Check if risk should be escalated based on regression history."""
        history = self._regression_history.get(domain.domain_id, [])

        if len(history) >= self.escalation_policy.regression_count_threshold:
            # Escalate risk level
            self.escalate_risk(domain.domain_id, domain)

    def escalate_risk(
        self,
        operator_or_domain: str,
        domain: Optional[BoundedDomain] = None,
    ) -> int:
        """
        Escalate risk level for a domain after repeated failures.

        Per Section 9.1 of rhizome_godel_alpha.md:
        "Risk level can only increase, never decrease automatically."

        Args:
            operator_or_domain: Domain ID or operator name
            domain: Optional domain object

        Returns:
            New risk level
        """
        if domain is None:
            domain = self.domains.get(operator_or_domain)

        if domain is None:
            return self.global_settings.default_risk

        old_risk = domain._current_risk
        new_risk = min(
            old_risk + 1,
            self.escalation_policy.max_escalation_level,
        )

        if new_risk > old_risk:
            domain._current_risk = new_risk

            # Emit escalation event
            self._emit_event(
                topic=self.bus_topics.risk_escalated,
                kind="metric",
                level="warn",
                data={
                    "domain_id": domain.domain_id,
                    "old_risk": old_risk,
                    "new_risk": new_risk,
                    "regression_count": len(self._regression_history.get(domain.domain_id, [])),
                    "max_risk": self.escalation_policy.max_escalation_level,
                },
            )

        return new_risk

    def validate_mutation(
        self,
        path: str,
        operator: str,
        risk_level: int,
    ) -> tuple[bool, list[str]]:
        """
        Validate a proposed mutation against domain rules.

        Returns:
            Tuple of (valid, list of violation messages)
        """
        violations: list[str] = []

        # Check domain containment
        domain = self.get_domain(path)
        if domain is None:
            violations.append(f"Path '{path}' not in any bounded domain")
            return False, violations

        # Check operator
        allowed, reason = self.check_operator_allowed(path, operator)
        if not allowed:
            violations.append(reason or "Operator not allowed")

        # Check risk level
        risk_allowed, effective_risk, risk_reason = self.check_risk_level(path, risk_level)
        if not risk_allowed:
            violations.append(risk_reason or "Risk level too high")

        # Emit domain check event
        self._emit_event(
            topic=self.bus_topics.domain_check,
            kind="metric",
            level="debug",
            data={
                "path": path,
                "operator": operator,
                "risk_level": risk_level,
                "domain_id": domain.domain_id if domain else None,
                "valid": len(violations) == 0,
                "violations": violations,
            },
        )

        return len(violations) == 0, violations


def load_domains(path: Optional[Path] = None) -> DomainManager:
    """
    Load domain configuration from YAML file.

    Args:
        path: Path to configuration file (defaults to /pluribus/.rgma-domains.yaml)

    Returns:
        Configured DomainManager instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Domain configuration not found: {path}")

    config = load_yaml(path)

    # Validate version
    version = config.get("version", "1.0")
    schema = config.get("schema", "rgma.domains.v1")
    if not schema.startswith("rgma.domains"):
        raise ValueError(f"Invalid schema: {schema}")

    # Parse global settings
    global_data = config.get("global", {})
    global_settings = GlobalSettings.from_dict(global_data)

    # Parse escalation policy
    escalation_data = config.get("escalation", {})
    escalation_policy = EscalationPolicy.from_dict(escalation_data)

    # Parse bus topics
    topics_data = config.get("bus_topics", {})
    bus_topics = BusTopics.from_dict(topics_data)

    # Parse domains
    domains_data = config.get("domains", [])
    domains: list[BoundedDomain] = []

    for domain_dict in domains_data:
        try:
            domain = BoundedDomain.from_dict(domain_dict)
            domains.append(domain)
        except Exception as e:
            raise ValueError(f"Invalid domain configuration: {e}")

    return DomainManager(
        domains=domains,
        global_settings=global_settings,
        escalation_policy=escalation_policy,
        bus_topics=bus_topics,
    )


# CLI Commands

def cmd_check(args: argparse.Namespace) -> int:
    """Check if a path is in a bounded domain."""
    try:
        manager = load_domains(Path(args.config) if args.config else None)
    except FileNotFoundError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1

    path = args.path
    domain = manager.get_domain(path)

    if domain is None:
        print(f"Path '{path}' is NOT in any bounded domain")
        print(f"Default risk level: {manager.global_settings.default_risk}")
        return 1

    print(f"Path '{path}' is in domain: {domain.domain_id}")
    print(f"  Description: {domain.description}")
    print(f"  Max Risk Level: {domain.max_risk_level}")
    print(f"  CMP Floor: {domain.cmp_floor}")
    print(f"  Rollback on Regression: {domain.rollback_on_regression}")
    print(f"  Allowed Operators: {', '.join(domain.allowed_operators)}")

    if args.json:
        result = {
            "path": path,
            "in_domain": True,
            "domain_id": domain.domain_id,
            "description": domain.description,
            "max_risk_level": domain.max_risk_level,
            "cmp_floor": domain.cmp_floor,
            "rollback_on_regression": domain.rollback_on_regression,
            "allowed_operators": domain.allowed_operators,
        }
        print(json.dumps(result, indent=2))

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all configured domains."""
    try:
        manager = load_domains(Path(args.config) if args.config else None)
    except FileNotFoundError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1

    domains = manager.list_domains()

    if args.json:
        result = [{
            "domain_id": d.domain_id,
            "description": d.description,
            "max_risk_level": d.max_risk_level,
            "cmp_floor": d.cmp_floor,
            "allowed_paths": d.allowed_paths,
            "allowed_operators": d.allowed_operators,
        } for d in domains]
        print(json.dumps(result, indent=2))
        return 0

    print(f"Configured RGMA Bounded Domains ({len(domains)}):")
    print("-" * 60)

    for domain in domains:
        print(f"\n{domain.domain_id}:")
        print(f"  {domain.description}")
        print(f"  Risk: {domain.max_risk_level}, CMP Floor: {domain.cmp_floor}")
        print(f"  Paths: {len(domain.allowed_paths)} allowed, {len(domain.forbidden_paths)} forbidden")
        print(f"  Operators: {', '.join(domain.allowed_operators)}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate domain configuration file."""
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH

    try:
        manager = load_domains(config_path)
        domains = manager.list_domains()

        print(f"Configuration valid: {config_path}")
        print(f"  Domains: {len(domains)}")
        print(f"  Global CMP Floor: {manager.global_settings.cmp_floor}")
        print(f"  Default Risk: {manager.global_settings.default_risk}")
        print(f"  Event Emission: {manager.global_settings.emit_events}")

        return 0

    except FileNotFoundError as e:
        sys.stderr.write(f"ERROR: Configuration file not found: {e}\n")
        return 1
    except ValueError as e:
        sys.stderr.write(f"ERROR: Invalid configuration: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"ERROR: Unexpected error: {e}\n")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed info about a specific domain."""
    try:
        manager = load_domains(Path(args.config) if args.config else None)
    except FileNotFoundError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1

    domain = manager.get_domain_by_id(args.domain_id)

    if domain is None:
        sys.stderr.write(f"ERROR: Domain '{args.domain_id}' not found\n")
        return 1

    if args.json:
        result = {
            "domain_id": domain.domain_id,
            "description": domain.description,
            "allowed_paths": domain.allowed_paths,
            "forbidden_paths": domain.forbidden_paths,
            "allowed_operators": domain.allowed_operators,
            "max_risk_level": domain.max_risk_level,
            "effective_risk": domain.get_effective_risk(),
            "cmp_floor": domain.cmp_floor,
            "rollback_on_regression": domain.rollback_on_regression,
            "metrics": {
                "track_mutations": domain.metrics.track_mutations,
                "alert_on_regression": domain.metrics.alert_on_regression,
                "regression_threshold": domain.metrics.regression_threshold,
            },
        }
        print(json.dumps(result, indent=2))
        return 0

    print(f"Domain: {domain.domain_id}")
    print("=" * 60)
    print(f"Description: {domain.description}")
    print()
    print("Allowed Paths:")
    for p in domain.allowed_paths:
        print(f"  + {p}")
    print()
    print("Forbidden Paths:")
    for p in domain.forbidden_paths:
        print(f"  - {p}")
    if not domain.forbidden_paths:
        print("  (none)")
    print()
    print("Allowed Operators:")
    for op in domain.allowed_operators:
        print(f"  * {op}")
    print()
    print(f"Risk Level: {domain.max_risk_level} (effective: {domain.get_effective_risk()})")
    print(f"CMP Floor: {domain.cmp_floor}")
    print(f"Rollback on Regression: {domain.rollback_on_regression}")
    print()
    print("Metrics:")
    print(f"  Track Mutations: {domain.metrics.track_mutations}")
    print(f"  Alert on Regression: {domain.metrics.alert_on_regression}")
    print(f"  Regression Threshold: {domain.metrics.regression_threshold:.1%}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="rgma_domains.py",
        description="RGMA Bounded Domain Management",
    )
    p.add_argument(
        "--config",
        default=None,
        help=f"Path to domain config file (default: {DEFAULT_CONFIG_PATH})",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # check command
    check = sub.add_parser("check", help="Check if a path is in a bounded domain")
    check.add_argument("path", help="File path to check")
    check.add_argument("--json", action="store_true", help="Output as JSON")
    check.set_defaults(func=cmd_check)

    # list command
    list_cmd = sub.add_parser("list", help="List all configured domains")
    list_cmd.add_argument("--json", action="store_true", help="Output as JSON")
    list_cmd.set_defaults(func=cmd_list)

    # validate command
    validate = sub.add_parser("validate", help="Validate domain configuration")
    validate.set_defaults(func=cmd_validate)

    # info command
    info = sub.add_parser("info", help="Show detailed info about a domain")
    info.add_argument("domain_id", help="Domain ID to inspect")
    info.add_argument("--json", action="store_true", help="Output as JSON")
    info.set_defaults(func=cmd_info)

    return p


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
