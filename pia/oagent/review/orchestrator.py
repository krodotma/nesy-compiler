#!/usr/bin/env python3
"""
Review Agent Orchestrator (Step 160)

Coordinates the full review pipeline, aggregating results from all analyzers.

PBTSO Phase: PLAN, DISTRIBUTE
Bus Topics: a2a.review.orchestrate, review.pipeline.complete

Pipeline stages:
1. Static Analysis (Step 152)
2. Security Scanning (Step 153)
3. Code Smell Detection (Step 154)
4. Architecture Checking (Step 155)
5. Documentation Checking (Step 156)
6. Comment Generation (Step 157)
7. Dependency Scanning (Step 158)
8. License Checking (Step 159)

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Import all review components
from .bootstrap import ReviewAgentBootstrap, ReviewAgentConfig, PBTSOPhase
from .static.engine import StaticAnalysisEngine, AnalysisResult
from .security.scanner import SecurityScanner, SecurityScanResult
from .smells.detector import CodeSmellDetector, SmellDetectionResult
from .architecture.checker import ArchitectureChecker, ArchitectureCheckResult
from .docs.checker import DocChecker, DocCheckResult
from .comments.generator import CommentGenerator, GeneratedReview, CommentSeverity
from .deps.vulnerability_scanner import DependencyScanner, DependencyScanResult
from .compliance.license_checker import LicenseChecker, LicenseCheckResult


# ============================================================================
# Types
# ============================================================================

class PipelineStage(Enum):
    """Stages in the review pipeline."""
    STATIC_ANALYSIS = "static_analysis"
    SECURITY_SCAN = "security_scan"
    SMELL_DETECTION = "smell_detection"
    ARCHITECTURE_CHECK = "architecture_check"
    DOC_CHECK = "doc_check"
    DEP_SCAN = "dep_scan"
    LICENSE_CHECK = "license_check"
    COMMENT_GENERATION = "comment_generation"


class ReviewDecision(Enum):
    """Final review decision."""
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    COMMENT = "comment"


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: PipelineStage
    success: bool
    duration_ms: float
    issue_count: int
    blocking_count: int
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["stage"] = self.stage.value
        return result


@dataclass
class ReviewResult:
    """Complete result from the review pipeline."""
    review_id: str
    files_reviewed: List[str]
    stages: List[StageResult] = field(default_factory=list)
    generated_review: Optional[GeneratedReview] = None
    decision: ReviewDecision = ReviewDecision.COMMENT
    total_issues: int = 0
    blocking_issues: int = 0
    duration_ms: float = 0
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "review_id": self.review_id,
            "files_reviewed": self.files_reviewed,
            "stages": [s.to_dict() for s in self.stages],
            "generated_review": self.generated_review.to_dict() if self.generated_review else None,
            "decision": self.decision.value,
            "total_issues": self.total_issues,
            "blocking_issues": self.blocking_issues,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class ReviewPipeline:
    """
    Configuration for the review pipeline.

    Attributes:
        enable_static: Enable static analysis
        enable_security: Enable security scanning
        enable_smells: Enable code smell detection
        enable_architecture: Enable architecture checking
        enable_docs: Enable documentation checking
        enable_deps: Enable dependency scanning
        enable_license: Enable license checking
        fail_on_security: Fail review on security issues
        fail_on_architecture: Fail review on architecture violations
        project_license: Project license SPDX ID
        timeout_seconds: Maximum pipeline duration
    """
    enable_static: bool = True
    enable_security: bool = True
    enable_smells: bool = True
    enable_architecture: bool = True
    enable_docs: bool = True
    enable_deps: bool = True
    enable_license: bool = True
    fail_on_security: bool = True
    fail_on_architecture: bool = True
    project_license: Optional[str] = None
    timeout_seconds: int = 600

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Review Orchestrator
# ============================================================================

class ReviewOrchestrator:
    """
    Orchestrates the full review pipeline.

    Coordinates multiple analyzers and produces a unified review.

    Example:
        config = ReviewAgentConfig()
        pipeline = ReviewPipeline()
        orchestrator = ReviewOrchestrator(config, pipeline)

        result = orchestrator.run(["/path/to/file.py"])
        print(result.generated_review.to_markdown())
    """

    def __init__(
        self,
        config: ReviewAgentConfig,
        pipeline: ReviewPipeline,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Review agent configuration
            pipeline: Pipeline configuration
            bus_path: Path to event bus file
        """
        self.config = config
        self.pipeline = pipeline
        self.bus_path = bus_path or self._get_bus_path()

        # Initialize bootstrap
        self.bootstrap = ReviewAgentBootstrap(config)

        # Initialize analyzers (lazy loaded)
        self._static_analyzer: Optional[StaticAnalysisEngine] = None
        self._security_scanner: Optional[SecurityScanner] = None
        self._smell_detector: Optional[CodeSmellDetector] = None
        self._architecture_checker: Optional[ArchitectureChecker] = None
        self._doc_checker: Optional[DocChecker] = None
        self._dep_scanner: Optional[DependencyScanner] = None
        self._license_checker: Optional[LicenseChecker] = None

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "orchestration",
            "actor": self.config.agent_id,
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _get_static_analyzer(self) -> StaticAnalysisEngine:
        """Get or create static analyzer."""
        if not self._static_analyzer:
            self._static_analyzer = StaticAnalysisEngine(
                timeout=self.config.timeout_seconds,
                bus_path=self.bus_path,
            )
        return self._static_analyzer

    def _get_security_scanner(self) -> SecurityScanner:
        """Get or create security scanner."""
        if not self._security_scanner:
            self._security_scanner = SecurityScanner(bus_path=self.bus_path)
        return self._security_scanner

    def _get_smell_detector(self) -> CodeSmellDetector:
        """Get or create smell detector."""
        if not self._smell_detector:
            self._smell_detector = CodeSmellDetector(bus_path=self.bus_path)
        return self._smell_detector

    def _get_architecture_checker(self) -> ArchitectureChecker:
        """Get or create architecture checker."""
        if not self._architecture_checker:
            self._architecture_checker = ArchitectureChecker(bus_path=self.bus_path)
        return self._architecture_checker

    def _get_doc_checker(self) -> DocChecker:
        """Get or create doc checker."""
        if not self._doc_checker:
            self._doc_checker = DocChecker(bus_path=self.bus_path)
        return self._doc_checker

    def _get_dep_scanner(self) -> DependencyScanner:
        """Get or create dependency scanner."""
        if not self._dep_scanner:
            self._dep_scanner = DependencyScanner(bus_path=self.bus_path)
        return self._dep_scanner

    def _get_license_checker(self) -> LicenseChecker:
        """Get or create license checker."""
        if not self._license_checker:
            self._license_checker = LicenseChecker(
                project_license=self.pipeline.project_license,
                bus_path=self.bus_path,
            )
        return self._license_checker

    def run(
        self,
        files: List[str],
        project_path: Optional[str] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> ReviewResult:
        """
        Run the full review pipeline.

        Args:
            files: List of file paths to review
            project_path: Project root path (for dep/license scanning)
            on_progress: Optional callback for progress updates

        Returns:
            ReviewResult with all findings

        Emits:
            a2a.review.orchestrate (start)
            review.pipeline.complete (completion)
        """
        review_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        started_at = datetime.now(timezone.utc).isoformat() + "Z"

        # Emit orchestration start
        self._emit_event("a2a.review.orchestrate", {
            "review_id": review_id,
            "files": files[:20],
            "file_count": len(files),
            "pipeline": self.pipeline.to_dict(),
            "status": "started",
        })

        result = ReviewResult(
            review_id=review_id,
            files_reviewed=files,
            started_at=started_at,
        )

        # Initialize comment generator
        comment_generator = CommentGenerator(bus_path=self.bus_path)

        # Track progress
        total_stages = sum([
            self.pipeline.enable_static,
            self.pipeline.enable_security,
            self.pipeline.enable_smells,
            self.pipeline.enable_architecture,
            self.pipeline.enable_docs,
            self.pipeline.enable_deps,
            self.pipeline.enable_license,
            True,  # Comment generation always runs
        ])
        completed_stages = 0

        def report_progress(stage: str):
            nonlocal completed_stages
            completed_stages += 1
            if on_progress:
                on_progress(stage, completed_stages / total_stages * 100)

        # Run pipeline stages
        try:
            # Stage 1: Static Analysis
            if self.pipeline.enable_static:
                stage_result = self._run_static_analysis(files)
                result.stages.append(stage_result)
                if stage_result.data and "issues" in stage_result.data:
                    comment_generator.add_static_issues(stage_result.data["issues"])
                report_progress("static_analysis")

            # Stage 2: Security Scan
            if self.pipeline.enable_security:
                stage_result = self._run_security_scan(files)
                result.stages.append(stage_result)
                if stage_result.data and "vulnerabilities" in stage_result.data:
                    comment_generator.add_security_vulnerabilities(
                        stage_result.data["vulnerabilities"]
                    )
                report_progress("security_scan")

            # Stage 3: Smell Detection
            if self.pipeline.enable_smells:
                stage_result = self._run_smell_detection(files)
                result.stages.append(stage_result)
                if stage_result.data and "smells" in stage_result.data:
                    comment_generator.add_code_smells(stage_result.data["smells"])
                report_progress("smell_detection")

            # Stage 4: Architecture Check
            if self.pipeline.enable_architecture:
                stage_result = self._run_architecture_check(files)
                result.stages.append(stage_result)
                if stage_result.data and "violations" in stage_result.data:
                    comment_generator.add_architecture_violations(
                        stage_result.data["violations"]
                    )
                report_progress("architecture_check")

            # Stage 5: Documentation Check
            if self.pipeline.enable_docs:
                stage_result = self._run_doc_check(files)
                result.stages.append(stage_result)
                if stage_result.data and "issues" in stage_result.data:
                    comment_generator.add_doc_issues(stage_result.data["issues"])
                report_progress("doc_check")

            # Stage 6: Dependency Scan
            if self.pipeline.enable_deps and project_path:
                stage_result = self._run_dep_scan(project_path)
                result.stages.append(stage_result)
                report_progress("dep_scan")

            # Stage 7: License Check
            if self.pipeline.enable_license and project_path:
                stage_result = self._run_license_check(project_path)
                result.stages.append(stage_result)
                report_progress("license_check")

            # Stage 8: Generate Comments
            generated_review = comment_generator.generate()
            result.generated_review = generated_review
            report_progress("comment_generation")

            # Calculate totals
            result.total_issues = len(generated_review.comments)
            result.blocking_issues = (
                generated_review.blocker_count +
                generated_review.critical_count
            )

            # Determine decision
            if result.blocking_issues > 0:
                result.decision = ReviewDecision.REQUEST_CHANGES
            elif result.total_issues > 0:
                result.decision = ReviewDecision.COMMENT
            else:
                result.decision = ReviewDecision.APPROVE

        except Exception as e:
            # Handle pipeline failure
            self._emit_event("a2a.review.orchestrate", {
                "review_id": review_id,
                "status": "failed",
                "error": str(e),
            })
            raise

        result.duration_ms = (time.time() - start_time) * 1000
        result.completed_at = datetime.now(timezone.utc).isoformat() + "Z"

        # Emit completion
        self._emit_event("review.pipeline.complete", {
            "review_id": review_id,
            "decision": result.decision.value,
            "total_issues": result.total_issues,
            "blocking_issues": result.blocking_issues,
            "duration_ms": result.duration_ms,
            "stages_completed": len(result.stages),
        })

        return result

    def _run_static_analysis(self, files: List[str]) -> StageResult:
        """Run static analysis stage."""
        start = time.time()
        try:
            analyzer = self._get_static_analyzer()
            analysis_result = analyzer.analyze(files)

            return StageResult(
                stage=PipelineStage.STATIC_ANALYSIS,
                success=True,
                duration_ms=(time.time() - start) * 1000,
                issue_count=len(analysis_result.issues),
                blocking_count=analysis_result.error_count,
                data={
                    "issues": analysis_result.issues,
                    "error_count": analysis_result.error_count,
                    "warning_count": analysis_result.warning_count,
                },
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.STATIC_ANALYSIS,
                success=False,
                duration_ms=(time.time() - start) * 1000,
                issue_count=0,
                blocking_count=0,
                error=str(e),
            )

    def _run_security_scan(self, files: List[str]) -> StageResult:
        """Run security scanning stage."""
        start = time.time()
        try:
            scanner = self._get_security_scanner()
            scan_result = scanner.scan(files)

            blocking = scan_result.critical_count + scan_result.high_count

            return StageResult(
                stage=PipelineStage.SECURITY_SCAN,
                success=True,
                duration_ms=(time.time() - start) * 1000,
                issue_count=len(scan_result.vulnerabilities),
                blocking_count=blocking if self.pipeline.fail_on_security else 0,
                data={
                    "vulnerabilities": scan_result.vulnerabilities,
                    "critical_count": scan_result.critical_count,
                    "high_count": scan_result.high_count,
                },
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.SECURITY_SCAN,
                success=False,
                duration_ms=(time.time() - start) * 1000,
                issue_count=0,
                blocking_count=0,
                error=str(e),
            )

    def _run_smell_detection(self, files: List[str]) -> StageResult:
        """Run code smell detection stage."""
        start = time.time()
        try:
            detector = self._get_smell_detector()
            result = detector.detect(files)

            return StageResult(
                stage=PipelineStage.SMELL_DETECTION,
                success=True,
                duration_ms=(time.time() - start) * 1000,
                issue_count=len(result.smells),
                blocking_count=result.high_count,
                data={
                    "smells": result.smells,
                    "high_count": result.high_count,
                    "smell_types": result.smell_type_counts,
                },
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.SMELL_DETECTION,
                success=False,
                duration_ms=(time.time() - start) * 1000,
                issue_count=0,
                blocking_count=0,
                error=str(e),
            )

    def _run_architecture_check(self, files: List[str]) -> StageResult:
        """Run architecture checking stage."""
        start = time.time()
        try:
            checker = self._get_architecture_checker()
            result = checker.check(files)

            blocking = result.critical_count + result.error_count

            return StageResult(
                stage=PipelineStage.ARCHITECTURE_CHECK,
                success=True,
                duration_ms=(time.time() - start) * 1000,
                issue_count=len(result.violations),
                blocking_count=blocking if self.pipeline.fail_on_architecture else 0,
                data={
                    "violations": result.violations,
                    "critical_count": result.critical_count,
                    "error_count": result.error_count,
                },
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.ARCHITECTURE_CHECK,
                success=False,
                duration_ms=(time.time() - start) * 1000,
                issue_count=0,
                blocking_count=0,
                error=str(e),
            )

    def _run_doc_check(self, files: List[str]) -> StageResult:
        """Run documentation checking stage."""
        start = time.time()
        try:
            checker = self._get_doc_checker()
            result = checker.check(files)

            return StageResult(
                stage=PipelineStage.DOC_CHECK,
                success=True,
                duration_ms=(time.time() - start) * 1000,
                issue_count=len(result.issues),
                blocking_count=result.error_count,
                data={
                    "issues": result.issues,
                    "coverage": result.coverage_stats,
                },
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.DOC_CHECK,
                success=False,
                duration_ms=(time.time() - start) * 1000,
                issue_count=0,
                blocking_count=0,
                error=str(e),
            )

    def _run_dep_scan(self, project_path: str) -> StageResult:
        """Run dependency scanning stage."""
        start = time.time()
        try:
            scanner = self._get_dep_scanner()
            result = scanner.scan(project_path)

            blocking = result.critical_count + result.high_count

            return StageResult(
                stage=PipelineStage.DEP_SCAN,
                success=True,
                duration_ms=(time.time() - start) * 1000,
                issue_count=len(result.vulnerabilities),
                blocking_count=blocking,
                data={
                    "dependencies_found": result.dependencies_found,
                    "vulnerability_count": len(result.vulnerabilities),
                },
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.DEP_SCAN,
                success=False,
                duration_ms=(time.time() - start) * 1000,
                issue_count=0,
                blocking_count=0,
                error=str(e),
            )

    def _run_license_check(self, project_path: str) -> StageResult:
        """Run license checking stage."""
        start = time.time()
        try:
            checker = self._get_license_checker()
            # In production, would extract dependencies first
            result = checker.check([])

            return StageResult(
                stage=PipelineStage.LICENSE_CHECK,
                success=True,
                duration_ms=(time.time() - start) * 1000,
                issue_count=len(result.violations),
                blocking_count=len(result.violations),
                data={
                    "compliance_level": result.compliance_level.value,
                    "licenses_found": result.licenses_found,
                },
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.LICENSE_CHECK,
                success=False,
                duration_ms=(time.time() - start) * 1000,
                issue_count=0,
                blocking_count=0,
                error=str(e),
            )


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Review Orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Agent Orchestrator (Step 160)")
    parser.add_argument("files", nargs="+", help="Files to review")
    parser.add_argument("--project", help="Project root path")
    parser.add_argument("--project-license", help="Project license SPDX ID")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")
    parser.add_argument("--no-security", action="store_true", help="Disable security scan")
    parser.add_argument("--no-smells", action="store_true", help="Disable smell detection")
    parser.add_argument("--no-docs", action="store_true", help="Disable doc checking")

    args = parser.parse_args()

    config = ReviewAgentConfig()
    pipeline = ReviewPipeline(
        enable_security=not args.no_security,
        enable_smells=not args.no_smells,
        enable_docs=not args.no_docs,
        project_license=args.project_license,
    )

    orchestrator = ReviewOrchestrator(config, pipeline)

    def on_progress(stage: str, pct: float):
        print(f"  [{pct:.0f}%] {stage}")

    print("Running review pipeline...")
    result = orchestrator.run(
        files=args.files,
        project_path=args.project,
        on_progress=on_progress,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.markdown and result.generated_review:
        print(result.generated_review.to_markdown())
    else:
        print(f"\nReview Complete")
        print(f"  Decision: {result.decision.value}")
        print(f"  Total Issues: {result.total_issues}")
        print(f"  Blocking Issues: {result.blocking_issues}")
        print(f"  Duration: {result.duration_ms:.1f}ms")
        print(f"\nStages:")
        for stage in result.stages:
            status = "OK" if stage.success else "FAIL"
            print(f"  {stage.stage.value}: {status} ({stage.issue_count} issues)")

    return 0 if result.decision == ReviewDecision.APPROVE else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
