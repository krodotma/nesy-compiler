#!/usr/bin/env python3
"""
lens_collimator.py - Provider Routing Logic

The "lens" that focuses and routes LLM requests to optimal providers.
Implements routing based on:
- Provider availability
- Rate limits
- Task complexity
- Agent affinity
"""

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from agent_bus import emit_bus_event, Topics


class RoutingStrategy(Enum):
    AFFINITY = "affinity"      # Route to agent's preferred provider
    ROUND_ROBIN = "round_robin"  # Distribute evenly
    PRIORITY = "priority"       # Use highest priority available
    FALLBACK = "fallback"       # Try primary, fall back on failure


@dataclass
class ProviderHealth:
    """Provider health tracking."""
    provider_id: str
    last_success: float = 0.0
    last_failure: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: int = 0
    rate_limit_remaining: int = 100
    
    @property
    def health_score(self) -> float:
        """Compute health score 0.0 - 1.0."""
        if self.success_count + self.failure_count == 0:
            return 0.5  # Unknown
        
        success_rate = self.success_count / (self.success_count + self.failure_count)
        recency_penalty = 0.0
        
        # Penalize recent failures
        if self.last_failure > self.last_success:
            age = time.time() - self.last_failure
            if age < 60:  # Within last minute
                recency_penalty = 0.3
            elif age < 300:  # Within 5 minutes
                recency_penalty = 0.1
        
        return max(0.0, min(1.0, success_rate - recency_penalty))


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    provider_id: str
    reason: str
    alternatives: list[str] = field(default_factory=list)
    confidence: float = 1.0


class LensCollimator:
    """
    Routes LLM requests to optimal providers.
    
    The "lens" metaphor: focuses scattered requests into coherent provider assignments.
    """
    
    # Agent → preferred provider affinity
    AGENT_AFFINITY = {
        "claude_architect": ["claude-opus", "claude-sonnet"],
        "codex_codemaster": ["claude-sonnet", "claude-opus"],
        "gemini_polymath": ["gemini-2", "claude-opus"],
        "qwen_visionary": ["qwen-plus", "gemini-2"],
        "superworker_d": ["claude-sonnet", "qwen-plus"],
    }
    
    # Provider priority (1 = highest)
    PROVIDER_PRIORITY = {
        "claude-opus": 1,
        "claude-sonnet": 2,
        "gemini-2": 2,
        "qwen-plus": 3,
    }
    
    def __init__(self):
        self.health: dict[str, ProviderHealth] = {
            "claude-opus": ProviderHealth("claude-opus"),
            "claude-sonnet": ProviderHealth("claude-sonnet"),
            "gemini-2": ProviderHealth("gemini-2"),
            "qwen-plus": ProviderHealth("qwen-plus"),
        }
        self.request_count: dict[str, int] = {p: 0 for p in self.health}
        self._rr_index = 0
    
    def record_success(self, provider_id: str, latency_ms: int):
        """Record successful request."""
        h = self.health.get(provider_id)
        if h:
            h.last_success = time.time()
            h.success_count += 1
            # Rolling average latency
            h.avg_latency_ms = int((h.avg_latency_ms * 0.9) + (latency_ms * 0.1))
    
    def record_failure(self, provider_id: str, error: str):
        """Record failed request."""
        h = self.health.get(provider_id)
        if h:
            h.last_failure = time.time()
            h.failure_count += 1
    
    def route(
        self,
        agent_id: str,
        strategy: RoutingStrategy = RoutingStrategy.AFFINITY,
        exclude: list[str] = None,
    ) -> RoutingDecision:
        """Route request to optimal provider."""
        exclude = exclude or []
        available = [p for p in self.health if p not in exclude]
        
        if not available:
            return RoutingDecision(
                provider_id="",
                reason="No providers available",
                confidence=0.0,
            )
        
        if strategy == RoutingStrategy.AFFINITY:
            return self._route_affinity(agent_id, available)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return self._route_round_robin(available)
        elif strategy == RoutingStrategy.PRIORITY:
            return self._route_priority(available)
        else:  # FALLBACK
            return self._route_fallback(agent_id, available)
    
    def _route_affinity(self, agent_id: str, available: list[str]) -> RoutingDecision:
        """Route based on agent affinity."""
        preferred = self.AGENT_AFFINITY.get(agent_id, ["claude-sonnet"])
        
        for p in preferred:
            if p in available:
                h = self.health[p]
                if h.health_score >= 0.5:
                    return RoutingDecision(
                        provider_id=p,
                        reason=f"Agent affinity: {agent_id} prefers {p}",
                        alternatives=[x for x in preferred if x != p and x in available],
                        confidence=h.health_score,
                    )
        
        # Fallback to priority if no healthy affinity
        return self._route_priority(available)
    
    def _route_round_robin(self, available: list[str]) -> RoutingDecision:
        """Distribute evenly across providers."""
        self._rr_index = (self._rr_index + 1) % len(available)
        provider = available[self._rr_index]
        
        return RoutingDecision(
            provider_id=provider,
            reason="Round-robin distribution",
            alternatives=[x for x in available if x != provider],
            confidence=self.health[provider].health_score,
        )
    
    def _route_priority(self, available: list[str]) -> RoutingDecision:
        """Route to highest priority available."""
        sorted_providers = sorted(
            available,
            key=lambda p: (self.PROVIDER_PRIORITY.get(p, 99), -self.health[p].health_score)
        )
        
        provider = sorted_providers[0]
        return RoutingDecision(
            provider_id=provider,
            reason=f"Priority routing: {provider} is highest priority available",
            alternatives=sorted_providers[1:],
            confidence=self.health[provider].health_score,
        )
    
    def _route_fallback(self, agent_id: str, available: list[str]) -> RoutingDecision:
        """Try affinity first, prepare fallback chain."""
        primary = self._route_affinity(agent_id, available)
        
        # Build fallback chain
        fallbacks = [p for p in available if p != primary.provider_id]
        primary.alternatives = sorted(
            fallbacks,
            key=lambda p: self.PROVIDER_PRIORITY.get(p, 99)
        )
        primary.reason = f"Primary: {primary.provider_id}, fallbacks: {primary.alternatives}"
        
        return primary
    
    def get_status(self) -> dict:
        """Get current routing status."""
        return {
            p: {
                "health_score": round(h.health_score, 2),
                "avg_latency_ms": h.avg_latency_ms,
                "success_count": h.success_count,
                "failure_count": h.failure_count,
            }
            for p, h in self.health.items()
        }


# Singleton instance
_collimator: Optional[LensCollimator] = None


def get_collimator() -> LensCollimator:
    """Get or create singleton collimator."""
    global _collimator
    if _collimator is None:
        _collimator = LensCollimator()
    return _collimator


def route_request(agent_id: str, strategy: str = "affinity") -> RoutingDecision:
    """Route a request for an agent."""
    collimator = get_collimator()
    return collimator.route(agent_id, RoutingStrategy(strategy))


if __name__ == "__main__":
    # Test routing
    collimator = LensCollimator()
    
    print("🔬 Lens Collimator Test")
    print("=" * 40)
    
    for agent in ["claude_architect", "codex_codemaster", "gemini_polymath"]:
        decision = collimator.route(agent)
        print(f"\n{agent}:")
        print(f"  → {decision.provider_id}")
        print(f"  Reason: {decision.reason}")
        print(f"  Alternatives: {decision.alternatives}")
