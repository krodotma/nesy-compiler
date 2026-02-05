#!/usr/bin/env python3
"""
Step 120: Test Orchestrator

Coordinates all test components for comprehensive test automation.

PBTSO Phase: PLAN, TEST, VERIFY, DISTRIBUTE
Bus Topics:
- a2a.test.orchestrate (subscribes)
- test.pipeline.start (emits)
- test.pipeline.complete (emits)

Dependencies: Steps 111-119 (All Test Components)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


# ============================================================================
# Constants
# ============================================================================

class PipelineStage(Enum):
    """Stages in the test pipeline."""
    IMPACT_ANALYSIS = "impact_analysis"
    PRIORITIZATION = "prioritization"
    FLAKY_DETECTION = "flaky_detection"
    TEST_EXECUTION = "test_execution"
    MUTATION_TESTING = "mutation_testing"
    REGRESSION_DETECTION = "regression_detection"
    REPORTING = "reporting"


class PipelineStatus(Enum):
    """Status of pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ABORTED = "aborted"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class StageResult:
    """Result of a pipeline stage."""
    stage: PipelineStage
    status: PipelineStatus
    started_at: float
    completed_at: Optional[float] = None
    output: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Get stage duration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class OrchestratorConfig:
    """
    Configuration for test orchestration.

    Attributes:
        enabled_stages: Stages to run in the pipeline
        parallel_execution: Run independent stages in parallel
        fail_fast: Stop pipeline on first failure
        mutation_testing_enabled: Include mutation testing
        regression_detection_enabled: Include regression detection
        output_dir: Directory for orchestration reports
        timeout_s: Overall pipeline timeout
        changed_files: List of changed files for impact analysis
        base_ref: Git base reference
        head_ref: Git head reference
    """
    enabled_stages: List[PipelineStage] = field(default_factory=lambda: [
        PipelineStage.IMPACT_ANALYSIS,
        PipelineStage.PRIORITIZATION,
        PipelineStage.TEST_EXECUTION,
        PipelineStage.REGRESSION_DETECTION,
        PipelineStage.REPORTING,
    ])
    parallel_execution: bool = False
    fail_fast: bool = True
    mutation_testing_enabled: bool = False
    regression_detection_enabled: bool = True
    flaky_detection_enabled: bool = True
    output_dir: str = ".pluribus/test-agent/orchestration"
    timeout_s: int = 3600  # 1 hour
    changed_files: List[str] = field(default_factory=list)
    base_ref: str = "HEAD~1"
    head_ref: str = "HEAD"
    test_paths: List[str] = field(default_factory=lambda: ["tests/"])
    coverage_threshold: float = 0.8
    mutation_threshold: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled_stages": [s.value for s in self.enabled_stages],
            "parallel_execution": self.parallel_execution,
            "fail_fast": self.fail_fast,
            "mutation_testing_enabled": self.mutation_testing_enabled,
            "regression_detection_enabled": self.regression_detection_enabled,
            "timeout_s": self.timeout_s,
            "test_paths": self.test_paths,
        }


@dataclass
class OrchestratorResult:
    """Complete result of orchestrated test run."""
    run_id: str
    config: OrchestratorConfig
    started_at: float
    completed_at: Optional[float] = None
    status: PipelineStatus = PipelineStatus.PENDING
    stage_results: List[StageResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Get total duration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at

    def get_stage_result(self, stage: PipelineStage) -> Optional[StageResult]:
        """Get result for a specific stage."""
        for result in self.stage_results:
            if result.stage == stage:
                return result
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "status": self.status.value,
            "stage_results": [s.to_dict() for s in self.stage_results],
            "summary": self.summary,
            "metadata": self.metadata,
        }


# ============================================================================
# Bus Interface
# ============================================================================

class OrchestratorBus:
    """Bus interface for test orchestration."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }
        try:
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(event_with_meta) + "\n")
        except IOError:
            pass


# ============================================================================
# Test Orchestrator
# ============================================================================

class TestOrchestrator:
    """
    Coordinates all test components for comprehensive test automation.

    Pipeline stages:
    1. Impact Analysis - Determine affected tests from changes
    2. Prioritization - Order tests by importance
    3. Flaky Detection - Identify/quarantine flaky tests
    4. Test Execution - Run prioritized tests
    5. Mutation Testing - Assess test quality (optional)
    6. Regression Detection - Identify regressions
    7. Reporting - Generate comprehensive report

    PBTSO Phase: PLAN, TEST, VERIFY, DISTRIBUTE
    Bus Topics: a2a.test.orchestrate, test.pipeline.start, test.pipeline.complete
    """

    BUS_TOPICS = {
        "orchestrate": "a2a.test.orchestrate",
        "pipeline_start": "test.pipeline.start",
        "pipeline_complete": "test.pipeline.complete",
        "stage_start": "test.stage.start",
        "stage_complete": "test.stage.complete",
    }

    def __init__(self, bus=None):
        """
        Initialize the test orchestrator.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus or OrchestratorBus()
        self._stage_handlers: Dict[PipelineStage, Callable] = {
            PipelineStage.IMPACT_ANALYSIS: self._run_impact_analysis,
            PipelineStage.PRIORITIZATION: self._run_prioritization,
            PipelineStage.FLAKY_DETECTION: self._run_flaky_detection,
            PipelineStage.TEST_EXECUTION: self._run_test_execution,
            PipelineStage.MUTATION_TESTING: self._run_mutation_testing,
            PipelineStage.REGRESSION_DETECTION: self._run_regression_detection,
            PipelineStage.REPORTING: self._run_reporting,
        }
        self._abort_flag = False
        self._pipeline_context: Dict[str, Any] = {}

    def run_pipeline(self, config: Optional[OrchestratorConfig] = None) -> OrchestratorResult:
        """
        Execute the complete test pipeline.

        Args:
            config: Orchestration configuration

        Returns:
            OrchestratorResult with complete results
        """
        config = config or OrchestratorConfig()
        run_id = str(uuid.uuid4())

        result = OrchestratorResult(
            run_id=run_id,
            config=config,
            started_at=time.time(),
            status=PipelineStatus.RUNNING,
        )

        self._abort_flag = False
        self._pipeline_context = {
            "run_id": run_id,
            "config": config,
            "selected_tests": [],
            "test_results": [],
            "coverage_data": {},
            "mutation_results": {},
            "regressions": [],
        }

        # Emit pipeline start
        self._emit_event("pipeline_start", {
            "run_id": run_id,
            "stages": [s.value for s in config.enabled_stages],
        })

        try:
            # Run each enabled stage
            for stage in config.enabled_stages:
                if self._abort_flag:
                    break

                # Skip disabled stages
                if stage == PipelineStage.MUTATION_TESTING and not config.mutation_testing_enabled:
                    continue
                if stage == PipelineStage.FLAKY_DETECTION and not config.flaky_detection_enabled:
                    continue

                stage_result = self._run_stage(stage, config)
                result.stage_results.append(stage_result)

                # Check for failure
                if stage_result.status == PipelineStatus.FAILED and config.fail_fast:
                    result.status = PipelineStatus.FAILED
                    break

            # Determine overall status
            if not self._abort_flag and result.status != PipelineStatus.FAILED:
                failed_stages = [s for s in result.stage_results if s.status == PipelineStatus.FAILED]
                if failed_stages:
                    result.status = PipelineStatus.FAILED
                else:
                    result.status = PipelineStatus.PASSED

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.metadata["error"] = str(e)

        result.completed_at = time.time()

        # Generate summary
        result.summary = self._generate_summary(result)

        # Emit pipeline complete
        self._emit_event("pipeline_complete", {
            "run_id": run_id,
            "status": result.status.value,
            "duration_s": result.duration_s,
            "summary": result.summary,
        })

        # Save report
        self._save_report(result, config.output_dir)

        return result

    def _run_stage(
        self,
        stage: PipelineStage,
        config: OrchestratorConfig,
    ) -> StageResult:
        """Run a single pipeline stage."""
        stage_result = StageResult(
            stage=stage,
            status=PipelineStatus.RUNNING,
            started_at=time.time(),
        )

        # Emit stage start
        self._emit_event("stage_start", {
            "run_id": self._pipeline_context["run_id"],
            "stage": stage.value,
        })

        try:
            handler = self._stage_handlers.get(stage)
            if handler:
                output = handler(config)
                stage_result.output = output
                stage_result.status = PipelineStatus.PASSED
            else:
                stage_result.status = PipelineStatus.FAILED
                stage_result.error_message = f"No handler for stage: {stage}"

        except Exception as e:
            stage_result.status = PipelineStatus.FAILED
            stage_result.error_message = str(e)

        stage_result.completed_at = time.time()

        # Emit stage complete
        self._emit_event("stage_complete", {
            "run_id": self._pipeline_context["run_id"],
            "stage": stage.value,
            "status": stage_result.status.value,
            "duration_s": stage_result.duration_s,
        })

        return stage_result

    def _run_impact_analysis(self, config: OrchestratorConfig) -> Dict[str, Any]:
        """Run impact analysis stage."""
        from .impact import ImpactAnalyzer, ImpactConfig

        analyzer = ImpactAnalyzer(bus=self.bus)
        impact_config = ImpactConfig(
            base_ref=config.base_ref,
            head_ref=config.head_ref,
        )

        result = analyzer.analyze(impact_config)

        # Store selected tests in context
        self._pipeline_context["selected_tests"] = result.selected_tests
        self._pipeline_context["impact_result"] = result

        return {
            "selected_tests": len(result.selected_tests),
            "total_tests": len(result.all_tests),
            "selection_ratio": result.selection_ratio,
        }

    def _run_prioritization(self, config: OrchestratorConfig) -> Dict[str, Any]:
        """Run test prioritization stage."""
        from .priority import TestPrioritizer, PriorityConfig

        prioritizer = TestPrioritizer(bus=self.bus)

        # Get tests from impact analysis or use all
        tests = self._pipeline_context.get("selected_tests", [])
        if not tests:
            tests = self._discover_all_tests(config.test_paths)

        priority_config = PriorityConfig(
            changed_files=config.changed_files,
        )

        result = prioritizer.prioritize(tests, priority_config)

        # Update context with prioritized order
        self._pipeline_context["prioritized_tests"] = result.get_test_order()

        return {
            "prioritized_count": result.selected_tests,
            "estimated_duration_s": result.estimated_duration_s,
        }

    def _run_flaky_detection(self, config: OrchestratorConfig) -> Dict[str, Any]:
        """Run flaky test detection stage."""
        from .flaky import FlakyDetector, FlakyConfig

        detector = FlakyDetector(bus=self.bus)

        # Get quarantined tests
        quarantined = detector.get_quarantined_tests()

        # Filter out quarantined tests from execution
        tests = self._pipeline_context.get("prioritized_tests", [])
        filtered = [t for t in tests if t not in quarantined]

        self._pipeline_context["tests_to_run"] = filtered
        self._pipeline_context["quarantined_tests"] = quarantined

        return {
            "quarantined_count": len(quarantined),
            "tests_to_run": len(filtered),
        }

    def _run_test_execution(self, config: OrchestratorConfig) -> Dict[str, Any]:
        """Run test execution stage."""
        from .runner.orchestrator import TestRunnerOrchestrator, TestRunConfig, RunnerType

        orchestrator = TestRunnerOrchestrator(bus=self.bus)

        tests = self._pipeline_context.get(
            "tests_to_run",
            self._pipeline_context.get("prioritized_tests", config.test_paths)
        )

        run_config = TestRunConfig(
            test_paths=tests if isinstance(tests, list) else [tests],
            runner_type=RunnerType.PYTEST,
            parallel=True,
            workers=4,
            collect_coverage=True,
        )

        result = orchestrator.run_tests(run_config)

        # Store results in context
        self._pipeline_context["test_results"] = [r.to_dict() for r in result.results]
        self._pipeline_context["test_run"] = result

        return {
            "total_tests": result.total_tests,
            "passed": result.passed,
            "failed": result.failed,
            "skipped": result.skipped,
            "duration_s": result.duration_s,
            "coverage_percent": result.coverage_percent,
        }

    def _run_mutation_testing(self, config: OrchestratorConfig) -> Dict[str, Any]:
        """Run mutation testing stage."""
        from .mutation import MutationEngine, MutationConfig

        engine = MutationEngine(bus=self.bus)

        mutation_config = MutationConfig(
            source_paths=config.test_paths,  # Would be source paths in real impl
            test_paths=config.test_paths,
            sample_ratio=0.1,  # Sample for performance
        )

        result = engine.run_mutation_testing(mutation_config)

        self._pipeline_context["mutation_results"] = result

        return {
            "mutation_score": result.mutation_score,
            "total_mutants": result.total_mutants,
            "killed": result.killed,
            "survived": result.survived,
        }

    def _run_regression_detection(self, config: OrchestratorConfig) -> Dict[str, Any]:
        """Run regression detection stage."""
        from .regression import RegressionDetector, RegressionConfig

        detector = RegressionDetector(bus=self.bus)

        test_results = self._pipeline_context.get("test_results", [])
        run_id = self._pipeline_context.get("run_id", "")

        result = detector.detect_regressions(
            test_results=test_results,
            run_id=run_id,
            commit_sha=config.head_ref,
        )

        self._pipeline_context["regressions"] = result.regressions

        return {
            "regressions_found": result.total_regressions,
            "new_failures": result.new_failures,
            "performance_regressions": result.performance_regressions,
            "has_critical": result.has_critical,
        }

    def _run_reporting(self, config: OrchestratorConfig) -> Dict[str, Any]:
        """Run reporting stage."""
        # Collect all results
        report_data = {
            "run_id": self._pipeline_context["run_id"],
            "timestamp": time.time(),
            "config": config.to_dict(),
            "impact_analysis": self._pipeline_context.get("impact_result"),
            "test_results": self._pipeline_context.get("test_results"),
            "mutation_results": self._pipeline_context.get("mutation_results"),
            "regressions": [r.to_dict() for r in self._pipeline_context.get("regressions", [])],
            "quarantined_tests": self._pipeline_context.get("quarantined_tests", []),
        }

        # Save comprehensive report
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / f"pipeline_report_{report_data['run_id']}.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return {
            "report_path": str(report_file),
        }

    def _discover_all_tests(self, test_paths: List[str]) -> List[str]:
        """Discover all tests in the given paths."""
        tests = []

        for path in test_paths:
            p = Path(path)
            if p.is_file():
                tests.append(str(p))
            elif p.is_dir():
                for f in p.rglob("test_*.py"):
                    tests.append(str(f))
                for f in p.rglob("*_test.py"):
                    tests.append(str(f))

        return tests

    def _generate_summary(self, result: OrchestratorResult) -> Dict[str, Any]:
        """Generate pipeline summary."""
        summary = {
            "status": result.status.value,
            "duration_s": result.duration_s,
            "stages_run": len(result.stage_results),
            "stages_passed": sum(1 for s in result.stage_results if s.status == PipelineStatus.PASSED),
            "stages_failed": sum(1 for s in result.stage_results if s.status == PipelineStatus.FAILED),
        }

        # Add test execution summary if available
        test_stage = result.get_stage_result(PipelineStage.TEST_EXECUTION)
        if test_stage and test_stage.output:
            summary["tests"] = test_stage.output

        # Add regression summary if available
        regression_stage = result.get_stage_result(PipelineStage.REGRESSION_DETECTION)
        if regression_stage and regression_stage.output:
            summary["regressions"] = regression_stage.output

        return summary

    def _save_report(self, result: OrchestratorResult, output_dir: str) -> None:
        """Save orchestration report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        report_file = output_path / f"orchestration_{result.run_id}.json"
        with open(report_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Generate markdown
        report_md = output_path / f"orchestration_{result.run_id}.md"
        with open(report_md, "w") as f:
            f.write(self._generate_markdown_report(result))

    def _generate_markdown_report(self, result: OrchestratorResult) -> str:
        """Generate markdown orchestration report."""
        status_icon = {
            PipelineStatus.PASSED: "[PASS]",
            PipelineStatus.FAILED: "[FAIL]",
            PipelineStatus.ABORTED: "[ABORT]",
        }.get(result.status, "[?]")

        lines = [
            "# Test Pipeline Report",
            "",
            f"**Run ID**: {result.run_id}",
            f"**Status**: {status_icon} {result.status.value}",
            f"**Duration**: {result.duration_s:.2f}s",
            f"**Date**: {datetime.fromtimestamp(result.started_at).isoformat()}",
            "",
            "## Summary",
            "",
        ]

        for key, value in result.summary.items():
            if isinstance(value, dict):
                lines.append(f"### {key.replace('_', ' ').title()}")
                for k, v in value.items():
                    lines.append(f"- **{k}**: {v}")
            else:
                lines.append(f"- **{key}**: {value}")

        lines.extend([
            "",
            "## Pipeline Stages",
            "",
            "| Stage | Status | Duration |",
            "|-------|--------|----------|",
        ])

        for stage in result.stage_results:
            status = stage.status.value
            duration = f"{stage.duration_s:.2f}s"
            lines.append(f"| {stage.stage.value} | {status} | {duration} |")

        # Add stage details
        for stage in result.stage_results:
            if stage.output:
                lines.extend([
                    "",
                    f"### {stage.stage.value.replace('_', ' ').title()}",
                    "",
                ])
                if isinstance(stage.output, dict):
                    for k, v in stage.output.items():
                        lines.append(f"- **{k}**: {v}")
                else:
                    lines.append(f"{stage.output}")

        return "\n".join(lines)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "orchestration",
            "actor": "test-agent",
            "data": data,
        })

    def abort(self) -> None:
        """Abort the running pipeline."""
        self._abort_flag = True

    async def run_pipeline_async(
        self,
        config: Optional[OrchestratorConfig] = None,
    ) -> OrchestratorResult:
        """Async version of pipeline execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run_pipeline, config)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Orchestrator")
    parser.add_argument("--tests", nargs="*", default=["tests/"], help="Test paths")
    parser.add_argument("--base", default="HEAD~1", help="Base git reference")
    parser.add_argument("--head", default="HEAD", help="Head git reference")
    parser.add_argument("--mutation", action="store_true", help="Enable mutation testing")
    parser.add_argument("--no-flaky", action="store_true", help="Disable flaky detection")
    parser.add_argument("--no-regression", action="store_true", help="Disable regression detection")
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/orchestration")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = OrchestratorConfig(
        test_paths=args.tests,
        base_ref=args.base,
        head_ref=args.head,
        mutation_testing_enabled=args.mutation,
        flaky_detection_enabled=not args.no_flaky,
        regression_detection_enabled=not args.no_regression,
        output_dir=args.output,
    )

    orchestrator = TestOrchestrator()
    result = orchestrator.run_pipeline(config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        status_icon = "[PASS]" if result.status == PipelineStatus.PASSED else "[FAIL]"

        print(f"\n{'='*60}")
        print(f"Test Pipeline {status_icon}")
        print(f"{'='*60}")
        print(f"Run ID: {result.run_id}")
        print(f"Duration: {result.duration_s:.2f}s")
        print()

        print("Stages:")
        for stage in result.stage_results:
            status = "[OK]" if stage.status == PipelineStatus.PASSED else "[FAIL]"
            print(f"  {status} {stage.stage.value}: {stage.duration_s:.2f}s")

        print()
        print("Summary:")
        for key, value in result.summary.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")

        print(f"\nReport: {config.output_dir}/")

        if result.status == PipelineStatus.FAILED:
            exit(1)


if __name__ == "__main__":
    main()
