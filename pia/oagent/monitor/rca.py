#!/usr/bin/env python3
"""
Monitor Root Cause Analyzer - Step 275

Automated root cause analysis for incidents and anomalies.

PBTSO Phase: RESEARCH

Bus Topics:
- monitor.rca.analyze (subscribed)
- monitor.rca.result (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class RCAStatus(Enum):
    """RCA analysis status."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class CauseCategory(Enum):
    """Root cause categories."""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    CAPACITY = "capacity"
    SECURITY = "security"
    HUMAN_ERROR = "human_error"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for RCA findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CERTAIN = "certain"


@dataclass
class Evidence:
    """Evidence supporting a root cause finding.

    Attributes:
        evidence_id: Unique evidence ID
        source: Evidence source
        evidence_type: Type of evidence
        description: Evidence description
        data: Evidence data
        weight: Evidence weight (0-1)
        timestamp: When evidence was collected
    """
    evidence_id: str
    source: str
    evidence_type: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "evidence_type": self.evidence_type,
            "description": self.description,
            "data": self.data,
            "weight": self.weight,
            "timestamp": self.timestamp,
        }


@dataclass
class CauseHypothesis:
    """A hypothesis about the root cause.

    Attributes:
        hypothesis_id: Unique hypothesis ID
        category: Cause category
        description: Hypothesis description
        evidence: Supporting evidence
        confidence: Confidence level
        score: Hypothesis score (0-1)
        remediation: Suggested remediation
    """
    hypothesis_id: str
    category: CauseCategory
    description: str
    evidence: List[Evidence] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    score: float = 0.0
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "category": self.category.value,
            "description": self.description,
            "evidence_count": len(self.evidence),
            "confidence": self.confidence.value,
            "score": self.score,
            "remediation": self.remediation,
        }


@dataclass
class RCAResult:
    """Result of root cause analysis.

    Attributes:
        analysis_id: Unique analysis ID
        incident_id: Related incident ID
        status: Analysis status
        start_time: Analysis start time
        end_time: Analysis end time
        hypotheses: Generated hypotheses
        primary_cause: Most likely root cause
        contributing_factors: Contributing factors
        impact_assessment: Impact analysis
        recommendations: Action recommendations
        timeline: Event timeline
    """
    analysis_id: str
    incident_id: str
    status: RCAStatus = RCAStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    hypotheses: List[CauseHypothesis] = field(default_factory=list)
    primary_cause: Optional[CauseHypothesis] = None
    contributing_factors: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "incident_id": self.incident_id,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": (self.end_time - self.start_time) if self.end_time and self.start_time else None,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "primary_cause": self.primary_cause.to_dict() if self.primary_cause else None,
            "contributing_factors": self.contributing_factors,
            "impact_assessment": self.impact_assessment,
            "recommendations": self.recommendations,
            "timeline": self.timeline,
        }


class RootCauseAnalyzer:
    """
    Automated root cause analysis.

    The analyzer:
    - Collects evidence from multiple sources
    - Generates and evaluates hypotheses
    - Identifies the most likely root cause
    - Provides remediation recommendations

    Example:
        analyzer = RootCauseAnalyzer()

        # Analyze an incident
        result = await analyzer.analyze_incident(incident_id="inc-123")

        # Get the primary cause
        if result.primary_cause:
            print(f"Root cause: {result.primary_cause.description}")
            print(f"Confidence: {result.primary_cause.confidence.value}")
    """

    BUS_TOPICS = {
        "analyze": "monitor.rca.analyze",
        "result": "monitor.rca.result",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        bus_dir: Optional[str] = None,
    ):
        """Initialize root cause analyzer.

        Args:
            bus_dir: Bus directory
        """
        self._analyses: Dict[str, RCAResult] = {}
        self._analysis_history: List[RCAResult] = []
        self._evidence_collectors: List[Callable] = []
        self._hypothesis_generators: List[Callable] = []
        self._last_heartbeat = time.time()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default components
        self._register_default_collectors()
        self._register_default_generators()

    def register_evidence_collector(
        self,
        collector: Callable[[str, Dict[str, Any]], List[Evidence]]
    ) -> None:
        """Register an evidence collector.

        Args:
            collector: Collector function
        """
        self._evidence_collectors.append(collector)

    def register_hypothesis_generator(
        self,
        generator: Callable[[List[Evidence]], List[CauseHypothesis]]
    ) -> None:
        """Register a hypothesis generator.

        Args:
            generator: Generator function
        """
        self._hypothesis_generators.append(generator)

    async def analyze_incident(
        self,
        incident_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RCAResult:
        """Analyze an incident to find root cause.

        Args:
            incident_id: Incident ID
            context: Additional context

        Returns:
            Analysis result
        """
        analysis_id = f"rca-{uuid.uuid4().hex[:8]}"

        result = RCAResult(
            analysis_id=analysis_id,
            incident_id=incident_id,
            status=RCAStatus.ANALYZING,
            start_time=time.time(),
        )

        self._analyses[analysis_id] = result

        try:
            # Phase 1: Collect evidence
            evidence = await self._collect_evidence(incident_id, context or {})

            # Phase 2: Build timeline
            result.timeline = self._build_timeline(evidence)

            # Phase 3: Generate hypotheses
            result.hypotheses = await self._generate_hypotheses(evidence)

            # Phase 4: Evaluate and rank hypotheses
            self._evaluate_hypotheses(result.hypotheses, evidence)

            # Phase 5: Select primary cause
            if result.hypotheses:
                result.hypotheses.sort(key=lambda h: -h.score)
                result.primary_cause = result.hypotheses[0]

            # Phase 6: Identify contributing factors
            result.contributing_factors = self._identify_contributing_factors(
                evidence, result.hypotheses
            )

            # Phase 7: Assess impact
            result.impact_assessment = self._assess_impact(evidence, context or {})

            # Phase 8: Generate recommendations
            result.recommendations = self._generate_recommendations(result)

            result.status = RCAStatus.COMPLETED

        except Exception as e:
            result.status = RCAStatus.FAILED
            result.recommendations = [f"Analysis failed: {str(e)}"]

        result.end_time = time.time()

        self._analysis_history.append(result)
        if len(self._analysis_history) > 100:
            self._analysis_history = self._analysis_history[-100:]

        self._emit_bus_event(
            self.BUS_TOPICS["result"],
            result.to_dict()
        )

        return result

    async def analyze_anomaly(
        self,
        anomaly_id: str,
        metric_name: str,
        anomaly_data: Dict[str, Any],
    ) -> RCAResult:
        """Analyze an anomaly to find root cause.

        Args:
            anomaly_id: Anomaly ID
            metric_name: Metric name
            anomaly_data: Anomaly data

        Returns:
            Analysis result
        """
        context = {
            "anomaly_id": anomaly_id,
            "metric_name": metric_name,
            "anomaly_data": anomaly_data,
        }

        return await self.analyze_incident(
            incident_id=f"anomaly:{anomaly_id}",
            context=context,
        )

    def get_analysis(self, analysis_id: str) -> Optional[RCAResult]:
        """Get an analysis by ID.

        Args:
            analysis_id: Analysis ID

        Returns:
            Analysis result or None
        """
        return self._analyses.get(analysis_id)

    def get_analyses_for_incident(self, incident_id: str) -> List[RCAResult]:
        """Get all analyses for an incident.

        Args:
            incident_id: Incident ID

        Returns:
            Analysis results
        """
        return [
            a for a in self._analyses.values()
            if a.incident_id == incident_id
        ]

    def get_recent_analyses(self, limit: int = 10) -> List[RCAResult]:
        """Get recent analyses.

        Args:
            limit: Maximum results

        Returns:
            Recent analyses
        """
        return list(reversed(self._analysis_history[-limit:]))

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics.

        Returns:
            Statistics
        """
        by_status: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        by_confidence: Dict[str, int] = {}

        for result in self._analysis_history:
            status = result.status.value
            by_status[status] = by_status.get(status, 0) + 1

            if result.primary_cause:
                cat = result.primary_cause.category.value
                by_category[cat] = by_category.get(cat, 0) + 1

                conf = result.primary_cause.confidence.value
                by_confidence[conf] = by_confidence.get(conf, 0) + 1

        avg_duration = 0
        completed = [r for r in self._analysis_history if r.end_time and r.start_time]
        if completed:
            avg_duration = sum(r.end_time - r.start_time for r in completed) / len(completed)

        return {
            "total_analyses": len(self._analysis_history),
            "by_status": by_status,
            "by_category": by_category,
            "by_confidence": by_confidence,
            "average_duration_s": avg_duration,
            "evidence_collectors": len(self._evidence_collectors),
            "hypothesis_generators": len(self._hypothesis_generators),
        }

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "rca",
                "status": "healthy",
                "analyses": len(self._analyses),
            }
        )

        return True

    async def _collect_evidence(
        self,
        incident_id: str,
        context: Dict[str, Any],
    ) -> List[Evidence]:
        """Collect evidence from all sources.

        Args:
            incident_id: Incident ID
            context: Analysis context

        Returns:
            Collected evidence
        """
        all_evidence: List[Evidence] = []

        for collector in self._evidence_collectors:
            try:
                evidence = collector(incident_id, context)
                all_evidence.extend(evidence)
            except Exception:
                pass

        return all_evidence

    async def _generate_hypotheses(
        self,
        evidence: List[Evidence],
    ) -> List[CauseHypothesis]:
        """Generate hypotheses from evidence.

        Args:
            evidence: Collected evidence

        Returns:
            Generated hypotheses
        """
        all_hypotheses: List[CauseHypothesis] = []

        for generator in self._hypothesis_generators:
            try:
                hypotheses = generator(evidence)
                all_hypotheses.extend(hypotheses)
            except Exception:
                pass

        return all_hypotheses

    def _evaluate_hypotheses(
        self,
        hypotheses: List[CauseHypothesis],
        evidence: List[Evidence],
    ) -> None:
        """Evaluate and score hypotheses.

        Args:
            hypotheses: Hypotheses to evaluate
            evidence: Available evidence
        """
        for hypothesis in hypotheses:
            # Calculate score based on evidence
            if not hypothesis.evidence:
                hypothesis.score = 0.1
                hypothesis.confidence = ConfidenceLevel.LOW
                continue

            # Weight by evidence quality
            total_weight = sum(e.weight for e in hypothesis.evidence)
            avg_weight = total_weight / len(hypothesis.evidence)

            # Factor in evidence count
            count_factor = min(1.0, len(hypothesis.evidence) / 5)

            hypothesis.score = avg_weight * 0.7 + count_factor * 0.3

            # Set confidence level
            if hypothesis.score >= 0.9:
                hypothesis.confidence = ConfidenceLevel.CERTAIN
            elif hypothesis.score >= 0.7:
                hypothesis.confidence = ConfidenceLevel.HIGH
            elif hypothesis.score >= 0.4:
                hypothesis.confidence = ConfidenceLevel.MEDIUM
            else:
                hypothesis.confidence = ConfidenceLevel.LOW

    def _build_timeline(self, evidence: List[Evidence]) -> List[Dict[str, Any]]:
        """Build event timeline from evidence.

        Args:
            evidence: Collected evidence

        Returns:
            Event timeline
        """
        events = []

        for e in evidence:
            events.append({
                "timestamp": e.timestamp,
                "source": e.source,
                "type": e.evidence_type,
                "description": e.description,
            })

        events.sort(key=lambda x: x["timestamp"])
        return events

    def _identify_contributing_factors(
        self,
        evidence: List[Evidence],
        hypotheses: List[CauseHypothesis],
    ) -> List[str]:
        """Identify contributing factors.

        Args:
            evidence: Collected evidence
            hypotheses: Generated hypotheses

        Returns:
            Contributing factors
        """
        factors: Set[str] = set()

        # Look for patterns in evidence
        sources = set(e.source for e in evidence)
        if len(sources) > 3:
            factors.add("Multiple systems affected - potential cascading failure")

        error_count = sum(1 for e in evidence if "error" in e.evidence_type.lower())
        if error_count > 5:
            factors.add(f"High error rate ({error_count} errors detected)")

        # Add secondary hypotheses as contributing factors
        for h in hypotheses[1:4]:  # Top 3 secondary hypotheses
            if h.score > 0.3:
                factors.add(f"Possible contribution: {h.description}")

        return list(factors)

    def _assess_impact(
        self,
        evidence: List[Evidence],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess incident impact.

        Args:
            evidence: Collected evidence
            context: Analysis context

        Returns:
            Impact assessment
        """
        affected_services: Set[str] = set()
        for e in evidence:
            affected_services.add(e.source)

        # Estimate severity based on evidence
        error_count = sum(1 for e in evidence if "error" in e.evidence_type.lower())
        critical_count = sum(1 for e in evidence if "critical" in e.data.get("severity", ""))

        if critical_count > 0:
            severity = "critical"
        elif error_count > 5:
            severity = "high"
        elif error_count > 0:
            severity = "medium"
        else:
            severity = "low"

        return {
            "affected_services": list(affected_services),
            "affected_count": len(affected_services),
            "severity": severity,
            "error_count": error_count,
            "critical_count": critical_count,
        }

    def _generate_recommendations(self, result: RCAResult) -> List[str]:
        """Generate remediation recommendations.

        Args:
            result: Analysis result

        Returns:
            Recommendations
        """
        recommendations: List[str] = []

        if not result.primary_cause:
            recommendations.append("Insufficient evidence for root cause determination")
            recommendations.append("Consider gathering more logs and metrics")
            return recommendations

        cause = result.primary_cause

        # Add remediation from hypothesis
        if cause.remediation:
            recommendations.append(cause.remediation)

        # Category-specific recommendations
        if cause.category == CauseCategory.INFRASTRUCTURE:
            recommendations.append("Review infrastructure health and capacity")
            recommendations.append("Check for hardware issues or network problems")

        elif cause.category == CauseCategory.APPLICATION:
            recommendations.append("Review application logs for stack traces")
            recommendations.append("Check for recent deployments or code changes")

        elif cause.category == CauseCategory.CONFIGURATION:
            recommendations.append("Audit recent configuration changes")
            recommendations.append("Validate configuration against known-good state")

        elif cause.category == CauseCategory.CAPACITY:
            recommendations.append("Review resource utilization metrics")
            recommendations.append("Consider scaling or capacity optimization")

        elif cause.category == CauseCategory.DEPENDENCY:
            recommendations.append("Check external service status")
            recommendations.append("Review dependency timeout and retry settings")

        # Add follow-up actions
        recommendations.append("Create postmortem document")
        recommendations.append("Update runbooks if applicable")

        return recommendations

    def _register_default_collectors(self) -> None:
        """Register default evidence collectors."""

        def log_collector(incident_id: str, context: Dict[str, Any]) -> List[Evidence]:
            """Collect evidence from logs."""
            evidence = []

            # Simulate log evidence
            evidence.append(Evidence(
                evidence_id=f"ev-{uuid.uuid4().hex[:8]}",
                source="log-collector",
                evidence_type="log_entry",
                description="Error detected in application logs",
                data={"level": "error"},
                weight=0.7,
            ))

            return evidence

        def metric_collector(incident_id: str, context: Dict[str, Any]) -> List[Evidence]:
            """Collect evidence from metrics."""
            evidence = []

            # Simulate metric evidence
            if context.get("anomaly_data"):
                evidence.append(Evidence(
                    evidence_id=f"ev-{uuid.uuid4().hex[:8]}",
                    source="metric-collector",
                    evidence_type="anomaly",
                    description=f"Anomaly detected in {context.get('metric_name', 'metric')}",
                    data=context.get("anomaly_data", {}),
                    weight=0.8,
                ))

            return evidence

        def change_collector(incident_id: str, context: Dict[str, Any]) -> List[Evidence]:
            """Collect evidence from change events."""
            evidence = []

            # Check for recent changes
            evidence.append(Evidence(
                evidence_id=f"ev-{uuid.uuid4().hex[:8]}",
                source="change-tracker",
                evidence_type="change_event",
                description="Configuration change detected",
                data={"change_type": "config"},
                weight=0.6,
            ))

            return evidence

        self._evidence_collectors.extend([
            log_collector,
            metric_collector,
            change_collector,
        ])

    def _register_default_generators(self) -> None:
        """Register default hypothesis generators."""

        def infrastructure_generator(evidence: List[Evidence]) -> List[CauseHypothesis]:
            """Generate infrastructure-related hypotheses."""
            hypotheses = []

            infra_evidence = [
                e for e in evidence
                if e.source in ("resource-monitor", "infrastructure")
                or "capacity" in e.evidence_type.lower()
            ]

            if infra_evidence:
                hypotheses.append(CauseHypothesis(
                    hypothesis_id=f"hyp-{uuid.uuid4().hex[:8]}",
                    category=CauseCategory.INFRASTRUCTURE,
                    description="Infrastructure resource constraint",
                    evidence=infra_evidence,
                    remediation="Scale resources or optimize resource usage",
                ))

            return hypotheses

        def application_generator(evidence: List[Evidence]) -> List[CauseHypothesis]:
            """Generate application-related hypotheses."""
            hypotheses = []

            app_evidence = [
                e for e in evidence
                if "error" in e.evidence_type.lower()
                or "exception" in e.evidence_type.lower()
            ]

            if app_evidence:
                hypotheses.append(CauseHypothesis(
                    hypothesis_id=f"hyp-{uuid.uuid4().hex[:8]}",
                    category=CauseCategory.APPLICATION,
                    description="Application error or exception",
                    evidence=app_evidence,
                    remediation="Review application logs and recent code changes",
                ))

            return hypotheses

        def config_generator(evidence: List[Evidence]) -> List[CauseHypothesis]:
            """Generate configuration-related hypotheses."""
            hypotheses = []

            config_evidence = [
                e for e in evidence
                if "config" in e.evidence_type.lower()
                or "change" in e.evidence_type.lower()
            ]

            if config_evidence:
                hypotheses.append(CauseHypothesis(
                    hypothesis_id=f"hyp-{uuid.uuid4().hex[:8]}",
                    category=CauseCategory.CONFIGURATION,
                    description="Configuration change caused issue",
                    evidence=config_evidence,
                    remediation="Rollback recent configuration changes",
                ))

            return hypotheses

        self._hypothesis_generators.extend([
            infrastructure_generator,
            application_generator,
            config_generator,
        ])

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_analyzer: Optional[RootCauseAnalyzer] = None


def get_analyzer() -> RootCauseAnalyzer:
    """Get or create the root cause analyzer singleton.

    Returns:
        RootCauseAnalyzer instance
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = RootCauseAnalyzer()
    return _analyzer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Root Cause Analyzer (Step 275)")
    parser.add_argument("--analyze", metavar="ID", help="Analyze incident")
    parser.add_argument("--show", metavar="ID", help="Show analysis result")
    parser.add_argument("--recent", action="store_true", help="Show recent analyses")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    analyzer = get_analyzer()

    if args.analyze:
        async def run():
            return await analyzer.analyze_incident(args.analyze)

        result = asyncio.run(run())
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Analysis: {result.analysis_id}")
            print(f"  Status: {result.status.value}")
            if result.primary_cause:
                print(f"  Primary Cause: {result.primary_cause.description}")
                print(f"  Category: {result.primary_cause.category.value}")
                print(f"  Confidence: {result.primary_cause.confidence.value}")
            print(f"  Recommendations:")
            for r in result.recommendations[:3]:
                print(f"    - {r}")

    if args.show:
        result = analyzer.get_analysis(args.show)
        if result:
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(f"Analysis: {result.analysis_id}")
                print(f"  Status: {result.status.value}")
                print(f"  Incident: {result.incident_id}")
        else:
            print(f"Analysis not found: {args.show}")

    if args.recent:
        analyses = analyzer.get_recent_analyses()
        if args.json:
            print(json.dumps([a.to_dict() for a in analyses], indent=2))
        else:
            print("Recent Analyses:")
            for a in analyses:
                cause = a.primary_cause.description if a.primary_cause else "Unknown"
                print(f"  [{a.status.value}] {a.analysis_id}: {cause}")

    if args.stats:
        stats = analyzer.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("RCA Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
