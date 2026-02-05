#!/usr/bin/env python3
"""
Test Agent - OAGENT Subagent 3 (Steps 101-150)

Automated test generation, mutation testing, and coverage analysis.

PBTSO Phases:
- SKILL: Core test generation capabilities
- SEQUESTER: Test isolation and sandboxing
- TEST: Test execution and result collection
- VERIFY: Coverage and mutation verification

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
- telemetry.test.coverage
"""

from .bootstrap import (
    TestAgentConfig,
    TestAgentBootstrap,
    PBTSOPhase,
)

__all__ = [
    "TestAgentConfig",
    "TestAgentBootstrap",
    "PBTSOPhase",
]

__version__ = "0.1.0"
__step_range__ = "101-110"
