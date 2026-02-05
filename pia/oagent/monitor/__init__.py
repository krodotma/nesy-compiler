#!/usr/bin/env python3
"""
Monitor Agent - OAGENT Subagent 6 (Steps 251-300)

Provides observability, metrics collection, anomaly detection, and alerting systems.

Architecture:
- Bootstrap: Agent initialization and A2A registration
- Metric Collector: Telemetry aggregation from all agents
- Metric Aggregator: Statistical rollups and windowed calculations
- Log Collector: Centralized log ingestion
- Log Analyzer: Pattern detection and classification
- Log Correlator: Cross-service correlation
- Anomaly Detector: Z-score and statistical anomaly detection
- Alert Router: Alert routing to appropriate channels
- Alert Manager: Alert lifecycle management
- Incident Automator: Automated incident response

PBTSO Phases:
- SKILL: Bootstrap initialization
- SEQUESTER: Security sandboxing for log access
- ITERATE: Metric/log collection and alert routing
- VERIFY: Anomaly detection and log analysis
- DISTILL: Metric aggregation
- RESEARCH: Log correlation

Bus Topics:
- a2a.monitor.bootstrap.start
- a2a.monitor.bootstrap.complete
- telemetry.*
- monitor.metrics.collected
- monitor.metrics.aggregate
- monitor.metrics.aggregated
- monitor.logs.collected
- monitor.logs.analyze
- monitor.logs.patterns
- monitor.logs.correlate
- monitor.logs.correlated
- monitor.anomaly.detect
- monitor.anomaly.detected
- monitor.alert.route
- monitor.alert.sent
- monitor.alert.create
- monitor.alert.acknowledge
- monitor.alert.resolve
- monitor.incident.create
- monitor.incident.escalate
- monitor.incident.resolve

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from .bootstrap import MonitorAgentBootstrap, MonitorAgentConfig

# Metrics (Steps 252-253)
from .metrics.collector import MetricCollector, MetricPoint, MetricType
from .metrics.aggregator import MetricAggregator, AggregatedMetric, AggregationType

# Logs (Steps 254-256)
from .logs.collector import LogCollector, LogEntry, LogLevel
from .logs.analyzer import LogAnalyzer, LogPattern, PatternType
from .logs.correlator import LogCorrelator, CorrelatedEvent, CorrelationChain

# Anomaly Detection (Step 257)
from .anomaly.detector import AnomalyDetector, Anomaly, AnomalySeverity, DetectionMethod

# Alerts (Steps 258-259)
from .alerts.router import AlertRouter, AlertRoute, AlertChannel, RoutingRule
from .alerts.manager import AlertManager, Alert, AlertState, AlertPriority

# Incidents (Step 260)
from .incident.automator import (
    IncidentAutomator,
    Incident,
    IncidentState,
    IncidentSeverity,
    ResponseAction,
    ResponsePlaybook,
)

__all__ = [
    # Bootstrap (Step 251)
    "MonitorAgentBootstrap",
    "MonitorAgentConfig",
    # Metrics (Steps 252-253)
    "MetricCollector",
    "MetricPoint",
    "MetricType",
    "MetricAggregator",
    "AggregatedMetric",
    "AggregationType",
    # Logs (Steps 254-256)
    "LogCollector",
    "LogEntry",
    "LogLevel",
    "LogAnalyzer",
    "LogPattern",
    "PatternType",
    "LogCorrelator",
    "CorrelatedEvent",
    "CorrelationChain",
    # Anomaly (Step 257)
    "AnomalyDetector",
    "Anomaly",
    "AnomalySeverity",
    "DetectionMethod",
    # Alerts (Steps 258-259)
    "AlertRouter",
    "AlertRoute",
    "AlertChannel",
    "RoutingRule",
    "AlertManager",
    "Alert",
    "AlertState",
    "AlertPriority",
    # Incidents (Step 260)
    "IncidentAutomator",
    "Incident",
    "IncidentState",
    "IncidentSeverity",
    "ResponseAction",
    "ResponsePlaybook",
]

__version__ = "0.1.0"
__step_range__ = "251-260"
